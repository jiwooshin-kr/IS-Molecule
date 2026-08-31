"""Reward-tilted posterior guidance for discrete diffusion (our method).

This file is self-contained on purpose: nothing in the original repository is
modified. The method is enabled with `guidance=ours` (see
`configs/guidance/ours.yaml`) and requires `OurGuidedDiffusion` instead of
`diffusion.Diffusion`.

Derivation being implemented
----------------------------
The reverse posterior marginalizes over the clean sequence,

    q(x_{t-1} | x_t) = E_{x_0 ~ q(x_0 | x_t)}[ q(x_{t-1} | x_t, x_0) ] ,

and we import-sample it with the model's denoiser p_theta(x_0 | x_t) as the
proposal. Because the forward noising process is shared,

    q(x_0 | x_t) / p_theta(x_0 | x_t)
        = [q(x_0) / p_theta(x_0)] * [p_theta(x_t) / q(x_t)] ,

and the second factor does not depend on x_0, so it cancels under
normalization. Defining the reward-tilted target

    q(x_0) \\propto p_theta(x_0) exp(lambda * r(x_0))

leaves the importance ratio equal to exp(lambda * r(x_0)), giving

    q(x_{t-1} | x_t) ~= sum_n w_n q(x_{t-1} | x_t, x_0^(n)),
    w_n = softmax_n(lambda * r(x_0^(n))),   x_0^(n) ~ p_theta(x_0 | x_t).

Consequences for implementation
-------------------------------
- The reward sees *clean* x_0 samples, so r can be any black-box function
  (RDKit QED / ring count here). Unlike D-CBG there is no classifier to train
  and no need for it to be robust to noised inputs.
- Cost per denoising step is one network forward (same as unguided sampling)
  plus N reward evaluations on CPU.
"""

import multiprocessing
import typing

import rdkit
import torch
import torch.nn.functional as F
import transformers
from rdkit import Chem as rdChem
from rdkit.Chem import QED
from tqdm.auto import tqdm

import classifier
import diffusion
from diffusion import _sample_categorical

rdkit.rdBase.DisableLog('rdApp.error')
rdkit.rdBase.DisableLog('rdApp.warning')

_SPECIAL_TOKEN_STRINGS = (
  '<bos>', '<eos>', '<pad>', '<mask>', '<unk>',
  '[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]',
)


def _mol_qed(mol: rdChem.Mol) -> float:
  return float(QED.qed(mol))


def _mol_ring_count(mol: rdChem.Mol) -> float:
  return float(len(rdChem.GetSymmSSSR(mol)))


REWARD_FNS: typing.Dict[str, typing.Callable[[rdChem.Mol], float]] = {
  'qed': _mol_qed,
  'ring_count': _mol_ring_count,
}


def clean_smiles(text: str) -> str:
  """Strips special tokens from a decoded sequence."""
  for token in _SPECIAL_TOKEN_STRINGS:
    text = text.replace(token, '')
  return text.strip()


def smiles_reward(
    smiles: str,
    reward_fn: typing.Callable[[rdChem.Mol], float],
    invalid_reward: float,
) -> float:
  """r(x_0) for a decoded SMILES string; `invalid_reward` if unparseable."""
  if not smiles:
    return invalid_reward
  try:
    mol = rdChem.MolFromSmiles(smiles)
  except Exception:  # RDKit raises several unrelated exception types
    return invalid_reward
  if mol is None:
    return invalid_reward
  try:
    return reward_fn(mol)
  except Exception:
    return invalid_reward


def smiles_is_valid(smiles: str) -> bool:
  """Can RDKit parse this? Parse only -- deliberately does not call the reward.

  This is the whole basis of oversampling: parsing costs ~0.04 ms while the
  reward (QED) costs ~0.81 ms, i.e. 91 % of the full evaluation. Screening 10x
  the candidates on validity and paying for the reward only on the ones that
  survive costs ~1.5x, not 10x. Reusing `smiles_reward` here would throw that
  away.
  """
  if not smiles:
    return False
  try:
    return rdChem.MolFromSmiles(smiles) is not None
  except Exception:
    return False


_WORKER = {}


def _init_reward_worker(reward_name: str, invalid_reward: float) -> None:
  rdkit.rdBase.DisableLog('rdApp.error')
  rdkit.rdBase.DisableLog('rdApp.warning')
  _WORKER['fn'] = REWARD_FNS[reward_name]
  _WORKER['invalid'] = invalid_reward


def _worker_reward(smiles: str) -> float:
  return smiles_reward(smiles, _WORKER['fn'], _WORKER['invalid'])


def _worker_valid(smiles: str) -> bool:
  # Runs on the same pool; needs none of the initialiser's state.
  return smiles_is_valid(smiles)


def _candidate_histograms(
    x0: torch.Tensor,
    weights: torch.Tensor,
    vocab_size: int,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
  """Reward-weighted and uniform-weight histograms of the same N candidates.

  Returns `(xbar_tilted, xbar_uniform)`, both (B, L, V):

    xbar_tilted  = sum_n w_n onehot(x_0^(n))     -- SNIS estimate of the tilted
                                                    denoiser marginal
    xbar_uniform = (1/N) sum_n onehot(x_0^(n))   -- plain MC estimate of
                                                    x_theta, whose expectation
                                                    is x_theta *exactly*

  The second one is what makes a control variate possible: it costs no extra
  reward evaluations (same candidates, already scored) and its mean is known in
  closed form, so subtracting its error is a zero-mean correction.

  Built with `scatter_add_` rather than `F.one_hot`, which would allocate an
  (N, B, L, V) tensor -- 650 MB at N=2000 on this vocabulary.
  """
  n, b, length = x0.shape
  idx = x0.permute(1, 2, 0)                       # (B, L, N)
  tilted = torch.zeros(b, length, vocab_size,
                       device=x0.device, dtype=weights.dtype)
  tilted.scatter_add_(2, idx, weights.permute(1, 0)[:, None, :]
                      .expand(b, length, n))
  uniform = torch.zeros_like(tilted)
  uniform.scatter_add_(2, idx, torch.full_like(tilted[..., :1], 1.0 / n)
                       .expand(b, length, n))
  return tilted, uniform


def _renorm_no_mask(dist: torch.Tensor, mask_index: int) -> torch.Tensor:
  """Drops the mask column and renormalises over the real tokens.

  Puts `x_theta` (already a distribution over the vocabulary) and `q_xs` (which
  carries the mask mass and is scaled by a per-position constant) onto the same
  footing so the two can be mixed. Renormalising cannot reorder anything on its
  own -- the divisor is constant within a position -- so an unmixed source
  ranks identically before and after.
  """
  out = dist.clone()
  out[:, :, mask_index] = 0.0
  return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class OurGuidedDiffusion(diffusion.Diffusion):
  """`diffusion.Diffusion` with the reward-tilted posterior of Eq. (8).

  Only the diffusion sampling loop is overridden; training, loss, and the
  baseline guidance methods are inherited untouched.
  """

  def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer):
    # Signature must match `Diffusion.__init__` exactly. Lightning's
    # `load_from_checkpoint` inspects it to decide which of its own kwargs to
    # forward; a `**kwargs` here makes it think everything is accepted and it
    # passes `logger=False` straight through to the parent, which rejects it.
    super().__init__(config, tokenizer=tokenizer)
    # SMILES -> reward, reused within a single `sample()` call. Candidates
    # repeat heavily once the sequence is mostly denoised.
    self._reward_cache: typing.Dict[str, float] = {}
    # Validity is cached separately: it is computed on the oversampled pool,
    # which is up to `oversample` times larger than the scored set.
    self._valid_cache: typing.Dict[str, bool] = {}
    self._last_pool_valid_frac: float = float('nan')
    self._last_short_frac: float = float('nan')
    # Fraction of sequences with no valid candidate at all, where the
    # validity-restricted target is empty and the step falls back to unguided.
    self._last_no_valid_frac: float = float('nan')
    # Fraction of sequences whose weights came out uniform, so the exact base
    # kernel replaced the Monte Carlo mixture.
    self._last_uniform_frac: float = float('nan')
    # Effective sample size of the importance weights, averaged over the
    # batch, for the most recent step. ESS ~= 1 means the mixture collapsed
    # onto a single candidate and lambda_ is too large for this reward scale.
    self._last_ess: float = float('nan')
    # Fraction of x_0 candidates RDKit could parse, and the fraction of
    # sequences for which *no* candidate parsed, at the most recent step.
    self._last_valid_frac: float = float('nan')
    self._last_all_invalid: float = float('nan')
    # Lazily built: token-id -> token string, with special ids mapped to ''.
    self._decode_table: typing.Optional[typing.List[str]] = None
    self._reward_pool: typing.Optional[multiprocessing.pool.Pool] = None

  # --- decoding ---------------------------------------------------------

  def _decode_smiles(self, ids: torch.Tensor) -> typing.List[str]:
    """Token ids of shape (M, L) to SMILES strings.

    `tokenizer.batch_decode` costs ~1.5 ms per sequence here, which at N x B
    candidates per step dominates the whole sampler (9.5 s per step at
    N=100, B=64). Joining against a precomputed table is ~560x faster and
    produces the same strings for this character-level SMILES vocabulary.
    """
    if self._decode_table is None:
      size = max(self.vocab_size, len(self.tokenizer))
      tokens = self.tokenizer.convert_ids_to_tokens(list(range(size)))
      special = set(self.tokenizer.all_special_ids)
      self._decode_table = [
        '' if i in special else (tokens[i] or '') for i in range(size)]
    table = self._decode_table
    return [''.join([table[i] for i in row])
            for row in ids.cpu().tolist()]

  def _get_reward_pool(self) -> typing.Optional[multiprocessing.pool.Pool]:
    workers = int(getattr(self.config.guidance, 'num_reward_workers', 0))
    if workers <= 1:
      return None
    if self._reward_pool is None:
      # Workers only touch RDKit on strings, never CUDA, so fork is safe.
      self._reward_pool = multiprocessing.get_context('fork').Pool(
        processes=workers,
        initializer=_init_reward_worker,
        initargs=(self.config.guidance.reward,
                  float(self.config.guidance.invalid_reward)))
    return self._reward_pool

  def close_reward_pool(self) -> None:
    if self._reward_pool is not None:
      self._reward_pool.close()
      self._reward_pool.join()
      self._reward_pool = None

  # --- reward / weights -----------------------------------------------

  # `_sample_categorical` materialises an (M, B, L, V) tensor; at M = 10 x 2000
  # that is 6.5 GB. Draw the candidate axis in chunks and keep only the int ids.
  _DRAW_CHUNK = 500

  def _validity(self, smiles_list: typing.List[str]) -> typing.List[bool]:
    """Parse-only validity, cached separately from the reward cache."""
    cache = self._valid_cache
    todo = list({s for s in smiles_list if s not in cache})
    if todo:
      pool = self._get_reward_pool()
      if pool is not None and len(todo) > 256:
        chunk = max(1, len(todo) // (pool._processes * 4))
        values = pool.map(_worker_valid, todo, chunksize=chunk)
      else:
        values = [smiles_is_valid(s) for s in todo]
      cache.update(zip(todo, values))
    return [cache[s] for s in smiles_list]

  def _draw_candidates(self, x_theta, n, b, length, vocab_size):
    """The N x_0 candidates to score, oversampling for validity if asked.

    With `guidance.oversample = k` this draws k*N candidates, screens them on
    validity (parse only, cheap), and keeps N of them: every valid one it can,
    in random order, then random invalid ones to pad. That makes the proposal
    p_theta(. | valid) rather than p_theta, which is a *different target* -- the
    same one `invalid_reward = -inf` defines -- but reached with N usable
    candidates instead of N x (valid fraction), which was measured as low as
    0.007 early in a UDLM trajectory.

    Note this is incompatible with `position_selection=cv_conf`: the control
    variate needs E[xbar_uniform] = x_theta exactly, which holds only when the
    candidates are drawn from x_theta itself.

    `oversample = 1` reduces to plain i.i.d. drawing, so it reproduces every
    earlier result.
    """
    over = int(getattr(self.config.guidance, 'oversample', 1) or 1)
    total = n * max(over, 1)
    chunks = []
    drawn = 0
    while drawn < total:
      size = min(self._DRAW_CHUNK, total - drawn)
      chunks.append(_sample_categorical(
        x_theta.unsqueeze(0).expand(size, b, length, vocab_size)))
      drawn += size
    x0 = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    if over <= 1:
      self._last_pool_valid_frac = float('nan')
      self._last_short_frac = float('nan')
      return x0

    smiles = [clean_smiles(t) for t in
              self._decode_smiles(x0.reshape(total * b, length))]
    valid = torch.tensor(self._validity(smiles), device=x0.device,
                         dtype=torch.float32).view(total, b)

    # Valid first in random order, then random invalids to pad -- one topk, no
    # loop over the batch. The 1e6 offset dominates any random key, so validity
    # strictly precedes the tie-break.
    key = torch.rand(total, b, device=x0.device) + 1e6 * valid
    idx = key.topk(n, dim=0).indices                       # (n, b)
    x0 = torch.gather(x0, 0, idx[:, :, None].expand(n, b, length))

    # Diagnostics: how good was the pool, and how often did it not contain N
    # valid candidates (the case where padding with invalids kicks in).
    self._last_pool_valid_frac = valid.mean().item()
    self._last_short_frac = (valid.sum(dim=0) < n).float().mean().item()
    return x0

  def _rewards(self, x0: torch.Tensor) -> torch.Tensor:
    """Rewards for x_0 candidates.

    Args:
      x0: candidate clean sequences of shape (N, B, L).

    Returns:
      Rewards of shape (N, B).
    """
    n, b, length = x0.shape
    smiles_list = [clean_smiles(text) for text in
                   self._decode_smiles(x0.reshape(n * b, length))]
    cache = self._reward_cache
    # RDKit costs ~1.3 ms per molecule (QED dominates), so score each distinct
    # string once and spread the misses over a process pool.
    todo = list({s for s in smiles_list if s not in cache})
    if todo:
      pool = self._get_reward_pool()
      if pool is not None and len(todo) > 256:
        chunk = max(1, len(todo) // (pool._processes * 4))
        values = pool.map(_worker_reward, todo, chunksize=chunk)
      else:
        reward_fn = REWARD_FNS[self.config.guidance.reward]
        invalid_reward = float(self.config.guidance.invalid_reward)
        values = [smiles_reward(s, reward_fn, invalid_reward) for s in todo]
      cache.update(zip(todo, values))
    rewards = torch.tensor(
      [cache[s] for s in smiles_list],
      device=x0.device, dtype=torch.float32).view(n, b)
    # Validity is returned alongside rather than inferred from the reward value:
    # `reward == invalid_reward` would misclassify a genuinely ring-free molecule
    # as unparseable under the ring_count reward. The validity cache makes this
    # free whenever the candidates were already screened, and a ~5 % parse
    # overhead otherwise (parsing is 0.04 ms against 0.81 ms for the reward).
    valid = torch.tensor(
      self._validity(smiles_list),
      device=x0.device, dtype=torch.bool).view(n, b)
    return rewards, valid

  def _mixture_weights(self, rewards: torch.Tensor,
                       valid: typing.Optional[torch.Tensor] = None
                       ) -> torch.Tensor:
    """w_n of Eq. (8); shape (N, B), summing to 1 over dim 0.

    With `guidance.exclude_invalid` the target is the tilted distribution
    *restricted to parseable molecules*, i.e. q(x_0) = 0 wherever RDKit cannot
    read x_0. Invalid candidates then carry zero weight and the estimator is a
    correct self-normalised IS estimate for that target: the proposal becomes
    p_theta(. | valid), and the resulting Z = P_theta(valid | x_t) does not
    depend on x_0, so it cancels in the normalisation and Eq. (8) is unchanged.

    This is done by masking the *logits*, never by putting -inf in the reward:
    `lambda_ * (-inf)` is NaN at lambda_ = 0, which is a grid point.

    When a sequence has no valid candidate at all the restricted target is empty
    and the weights are 0/0. Those columns fall back to uniform, i.e. that step
    is unguided for that sequence -- which is exactly what the unrestricted
    target does there too, since all its rewards are then equal.
    """
    if self.config.guidance.normalize_reward:
      rewards = ((rewards - rewards.mean(dim=0, keepdim=True))
                 / (rewards.std(dim=0, keepdim=True) + 1e-6))
    logits = float(self.config.guidance.lambda_) * rewards
    if valid is not None and bool(
        getattr(self.config.guidance, 'exclude_invalid', False)):
      none_valid = ~valid.any(dim=0, keepdim=True)     # (1, B)
      logits = logits.masked_fill(~valid & ~none_valid, float('-inf'))
      self._last_no_valid_frac = none_valid.float().mean().item()
    return torch.softmax(logits, dim=0)

  # --- posterior conditioned on a *given* x_0 -------------------------

  def _exact_marginal_posterior(
      self,
      x_theta: torch.Tensor,
      xt: torch.Tensor,
      move_chance_t: torch.Tensor,
      move_chance_s: torch.Tensor,
  ) -> torch.Tensor:
    """E_{x_0 ~ x_theta}[ q(x_{t-1} | x_t, x_0) ], exactly, with no sampling.

    This is the quantity Eq. (1)-(3) asks for at lambda_ = 0, where
    q(x_0 | x_t) = p_theta(x_0 | x_t). It is *not* what unguided sampling
    computes: upstream substitutes the soft x_theta into the posterior
    expression, which for uniform diffusion is a ratio-of-averages rather than
    the average-of-ratios, because the normalizer depends on x_0. The gap is not
    small -- measured up to 0.248 in total variation at t = 0.1.

    Absorbing state: the expression is linear in x_0, so plugging in x_theta
    already *is* the expectation.

    Uniform: the sum over the V possible clean tokens collapses, because the
    normalizer alpha_t V x_0[x_t] + (1 - alpha_t) takes only two values --- one
    when the candidate token equals x_t, one otherwise. Writing p = x_theta[x_t]
    and splitting the sum on that condition gives a closed form with no V-fold
    tensor. Verified against the brute-force V-term sum to machine precision.
    """
    if self.diffusion == 'absorbing_state':
      return self._posterior_from_x0_dist(
        x_theta, xt, move_chance_t, move_chance_s)
    if self.diffusion != 'uniform':
      raise NotImplementedError(
        f"Diffusion type {self.diffusion} not implemented.")
    vocab = self.vocab_size
    alpha_t = 1 - move_chance_t
    alpha_s = 1 - move_chance_s
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    const_num = (1 - alpha_ts) * (1 - alpha_s) / vocab
    denom_eq = alpha_t * vocab + (1 - alpha_t)     # candidate token == x_t
    denom_ne = 1 - alpha_t                         # candidate token != x_t
    p = torch.gather(x_theta, -1, xt[..., None])    # (B, L, 1)
    xt_oh = F.one_hot(xt, vocab).to(x_theta.dtype)
    at_xt = ((p / denom_eq) * (alpha_t * vocab + alpha_ts - alpha_t + d_alpha)
             + ((1 - p) / denom_ne) * (alpha_ts - alpha_t))
    const = (p / denom_eq + (1 - p) / denom_ne) * const_num
    return const + xt_oh * at_xt + (d_alpha / denom_ne) * x_theta * (1 - xt_oh)

  def _posterior_from_x0_dist(
      self,
      x0_dist: torch.Tensor,
      xt: torch.Tensor,
      move_chance_t: torch.Tensor,
      move_chance_s: torch.Tensor,
  ) -> torch.Tensor:
    """The posterior expression with `x0_dist` substituted for x_0.

    With a one-hot `x0_dist` this is the *exact* q(x_{t-1} | x_t, x_0). With a
    soft distribution it is the same expression unguided sampling uses when it
    plugs in the denoiser marginal x_theta, which for uniform diffusion is an
    approximation (the expression is nonlinear in x through its normalizer).

    Args:
      x0_dist: distribution over clean tokens, shape (B, L, V).
      xt: current latents of shape (B, L).
      move_chance_t: 1 - alpha_t, shape (B, 1, 1).
      move_chance_s: 1 - alpha_s, shape (B, 1, 1).

    Returns:
      Posterior of shape (B, L, V).
    """
    if self.diffusion == 'absorbing_state':
      q_xs = x0_dist * (move_chance_t - move_chance_s)
      q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
      return q_xs / move_chance_t
    if self.diffusion == 'uniform':
      return self._compute_posterior(
        x=x0_dist,
        xt=xt,
        alpha_s=1 - move_chance_s,
        alpha_t=1 - move_chance_t)
    raise NotImplementedError(
      f"Diffusion type {self.diffusion} not implemented.")

  def _posterior_given_x0(
      self,
      x0: torch.Tensor,
      xt: torch.Tensor,
      move_chance_t: torch.Tensor,
      move_chance_s: torch.Tensor,
  ) -> torch.Tensor:
    """Exact q(x_{t-1} | x_t, x_0) for known clean sequences x0 of shape (B, L)."""
    return self._posterior_from_x0_dist(
      F.one_hot(x0, self.vocab_size).to(move_chance_t.dtype),
      xt, move_chance_t, move_chance_s)

  # --- one guided denoising step ---------------------------------------

  def _our_denoise(
      self,
      xt: torch.Tensor,
      time_conditioning: torch.Tensor,
      move_chance_t: torch.Tensor,
      move_chance_s: torch.Tensor,
      t: float,
      cache: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
  ) -> typing.Tuple[torch.Tensor, torch.Tensor,
                    typing.Dict[str, torch.Tensor]]:
    # p_theta(x_0 | x_t): the single network forward for this step.
    if cache is not None:
      log_x_theta = cache['log_x_theta']
    else:
      log_x_theta = self.forward(xt, time_conditioning, cond=None)
      if self.config.sampling.use_float64:
        log_x_theta = log_x_theta.to(torch.float64)
    x_theta = log_x_theta.exp()
    b, length, vocab_size = x_theta.shape
    n = int(self.config.guidance.num_x0_samples)

    # Outside the guidance window fall back to the base sampler: plug the
    # denoiser marginal straight into the posterior. This is exactly
    # `Diffusion._ddpm_denoise` and skips the N reward evaluations.
    if not (float(self.config.guidance.t_min)
            <= t <= float(self.config.guidance.t_max)):
      self._last_ess = float('nan')
      q_xs = self._posterior_from_x0_dist(
        x_theta, xt, move_chance_t, move_chance_s)
      xs = _sample_categorical(q_xs)
      if self.diffusion == 'absorbing_state':
        copy_flag = (xt != self.mask_index).to(torch.bool)
        q_xs[copy_flag] = 0.0
        q_xs[copy_flag, xt[copy_flag]] = 1.0
        xs = torch.where(copy_flag, xt, xs)
      return xs, q_xs, {'log_x_theta': log_x_theta}

    # Eq. (4): N i.i.d. proposals x_0^(n) ~ p_theta(x_0 | x_t).
    # For absorbing-state diffusion the `subs` parameterization already pins
    # x_theta to one-hot on unmasked positions, so candidates agree with x_t
    # wherever x_t is not masked.
    x0 = self._draw_candidates(x_theta, n, b, length, vocab_size)

    # Eq. (8): w_n \propto exp(lambda * r(x_0^(n))).
    rewards, valid = self._rewards(x0)
    weights = self._mixture_weights(rewards, valid)
    # Effective sample size: N means the weights are uniform, 1 means they
    # collapsed onto a single candidate (lambda_ too large for this reward).
    self._last_ess = (1.0 / weights.pow(2).sum(dim=0)).mean().item()
    # Diagnostic for the "score only the valid candidates" question: a hard
    # validity constraint means renormalising over the valid subset, which is
    # undefined when a sequence has no valid candidate at all. Both numbers are
    # needed to know whether that fallback is rare or routine. RDKit essentially
    # never returns exactly `invalid_reward` for a parseable molecule, so
    # equality is a safe proxy for "unparseable".
    self._last_valid_frac = valid.float().mean().item()
    self._last_all_invalid = (~valid).all(dim=0).float().mean().item()

    hist = None
    mixture_sampling = self.config.guidance.mixture_sampling
    if mixture_sampling == 'edlm':
      # EDLM's Algorithm 1 (Denoising via Importance Sampling), with its learned
      # energy E_phi(x_0, x_t) replaced by our reward: sample the mixture
      # ancestrally -- draw the component, then draw from it. This keeps the
      # cross-position correlations of the joint mixture. Note those
      # correlations come *only* from the reward reweighting: the candidates
      # themselves are drawn position-independently from x_theta above, so at
      # lambda_ = 0 there is nothing left to preserve.
      component = _sample_categorical(weights.t())  # (B,)
      x0_selected = torch.gather(
        x0, 0,
        component[None, :, None].expand(1, b, length)).squeeze(0)
      q_xs = self._posterior_given_x0(
        x0_selected, xt, move_chance_t, move_chance_s)
    elif mixture_sampling == 'marginal':
      # Our method. Average the N *individually normalized* posteriors and
      # sample positions independently, which matches the mixture's per-position
      # marginals exactly and drops correlations.
      #
      # Normalizing each component before aggregating is what makes this the
      # faithful form of Eq. (8). For uniform diffusion the posterior normalizer
      # x_0 Qbar_t x_t^T depends on x_0, so aggregating first and dividing once
      # at the end computes a ratio of averages instead of an average of ratios
      # and is biased -- it systematically under-weights candidates that
      # disagree with x_t. For absorbing-state diffusion the normalizer is
      # 1 - alpha_t whatever x_0 is, so the two coincide there.
      xt_rep = xt.unsqueeze(0).expand(n, b, length).reshape(n * b, length)
      move_chance_t_rep = move_chance_t.unsqueeze(0).expand(
        n, b, 1, 1).reshape(n * b, 1, 1)
      move_chance_s_rep = move_chance_s.unsqueeze(0).expand(
        n, b, 1, 1).reshape(n * b, 1, 1)
      q_all = self._posterior_given_x0(
        x0.reshape(n * b, length), xt_rep,
        move_chance_t_rep, move_chance_s_rep).view(
        n, b, length, vocab_size)
      q_xs = (weights[:, :, None, None] * q_all).sum(dim=0)
      # Only needed by the control-variate position rule; cheap enough to skip
      # unless asked for.
      if str(getattr(self.config.guidance,
                     'position_selection', 'bernoulli')) == 'cv_conf':
        hist = _candidate_histograms(x0, weights, vocab_size)
    else:
      raise NotImplementedError(
        f"mixture_sampling={mixture_sampling} not implemented; expected "
        "'marginal' (ours) or 'edlm'.")

    # Where the weights came out uniform, the mixture is only a Monte Carlo
    # estimate of something we already know exactly.
    #
    # With w_n = 1/N the mixture is (1/N) sum_n q(. | x_t, x_0^(n)) with
    # x_0^(n) ~ p_theta(. | x_t) -- an unbiased but noisy estimate of
    # E_{x_0}[q(. | x_t, x_0)] = p_theta(x_{t-1} | x_t). For absorbing-state
    # diffusion the posterior is *linear* in x_0, so plugging in the denoiser
    # marginal x_theta gives that expectation in closed form, exactly. Using the
    # N-sample estimate instead adds variance and no information.
    #
    # Not a rare corner: the weights are uniform whenever lambda_ = 0, and also
    # whenever every candidate ties -- which is what happens when none of them
    # parses, measured at 31 % of sequences per step on MDLM at N=300.
    #
    # Exact for both diffusions, via `_exact_marginal_posterior`. For uniform
    # diffusion this is NOT upstream's kernel: the posterior is nonlinear in x_0
    # through its normalizer, so substituting x_theta gives a ratio of averages
    # instead of an average of ratios. The correct expectation has a closed form
    # anyway, because the normalizer takes only two values. Off by default so
    # results recorded earlier stay reproducible.
    if bool(getattr(self.config.guidance, 'exact_uniform_step', False)):
      spread = weights.max(dim=0).values - weights.min(dim=0).values   # (B,)
      is_uniform = spread < 1e-9
      self._last_uniform_frac = is_uniform.float().mean().item()
      if bool(is_uniform.any()):
        q_base = self._exact_marginal_posterior(
          x_theta, xt, move_chance_t, move_chance_s)
        q_xs = torch.where(is_uniform[:, None, None], q_base, q_xs)

    arm = str(getattr(self.config.guidance, 'position_selection', 'bernoulli'))
    # Position selection gets its own time window, independent of the guidance
    # window (`t_min`/`t_max`) above: the reward tilt and the choice of which
    # position to unmask are separate mechanisms and there is no reason for them
    # to switch on together.
    #
    # The motivation is the measured failure mode. Introducing a confidence
    # ranking at all costs 43 novel molecules out of ~360 (paired, p<0.05) --
    # ranking commits to the most confident, hence most typical, positions and
    # walks towards the mode. Early unmasking decisions fix the molecular
    # scaffold and are where novelty is decided; late ones are local
    # substitutions. Restricting the ranking to late t therefore asks whether the
    # quality gain survives without the diversity cost.
    #
    # Same convention as the guidance window: t runs 1 (noise) -> 0 (clean), so
    # `position_t_max < 1` means "late steps only".
    p_lo = float(getattr(self.config.guidance, 'position_t_min', 0.0) or 0.0)
    p_hi = getattr(self.config.guidance, 'position_t_max', 1.0)
    p_hi = 1.0 if p_hi is None else float(p_hi)
    if not (p_lo <= t <= p_hi):
      arm = 'bernoulli'
    if arm == 'bernoulli':
      # The schedule's own coin flip, unchanged. This is the default and the
      # only path that existed before position selection was added.
      xs = _sample_categorical(q_xs)
      if self.diffusion == 'absorbing_state':
        copy_flag = (xt != self.mask_index).to(torch.bool)
        q_xs[copy_flag] = 0.0
        q_xs[copy_flag, xt[copy_flag]] = 1.0
        xs = torch.where(copy_flag, xt, xs)
    else:
      if self.diffusion != 'absorbing_state':
        raise NotImplementedError(
          f"position_selection={arm} needs masked positions to choose between, "
          "which only absorbing-state diffusion has. Uniform diffusion "
          "resamples every position at every step, so there is nothing to rank.")
      xs = self._select_and_unmask(q_xs, x_theta, xt, move_chance_t,
                                   move_chance_s, arm, hist)
      copy_flag = (xt != self.mask_index).to(torch.bool)
      q_xs[copy_flag] = 0.0
      q_xs[copy_flag, xt[copy_flag]] = 1.0
    return xs, q_xs, {'log_x_theta': log_x_theta}

  # Jitter used to break ties in the position ranking. Small enough not to
  # reorder genuinely different confidences (which differ by >> 1e-6 in
  # practice), large enough to randomise exact ties.
  _TIE_EPS = 1e-6

  def _select_and_unmask(
      self,
      q_xs: torch.Tensor,
      x_theta: torch.Tensor,
      xt: torch.Tensor,
      move_chance_t: torch.Tensor,
      move_chance_s: torch.Tensor,
      arm: str,
      hist: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None,
  ) -> torch.Tensor:
    """Reward-aware confidence unmasking: choose *which* positions to unmask.

    In the base absorbing-state sampler the posterior at a masked position is
    `P(stay masked) = move_chance_s / move_chance_t`, the same value at every
    masked position and independent of the token distribution -- so the choice
    of which position to unmask carries no information at all. That is the slack
    this exploits: unmask where the reward-tilted candidate ensemble agrees most.

    Four arms, of which the last is ours; the first three exist to strip
    confounds off it:

      bernoulli    the schedule's coin flip (handled by the caller, not here)
      random_k     deterministic count, uniformly random positions
                   -> isolates the effect of fixing the *count*
      xtheta_conf  deterministic count, ranked by the denoiser marginal
                   -> isolates the effect of ranking *at all*
      xbar_conf    deterministic count, ranked by the reward-weighted candidate
                   histogram xbar_0                     -> ours; the contribution
                   is exactly xtheta_conf -> xbar_conf

    Ranking `xbar_conf` on `q_xs` is the same as ranking on xbar_0: at a masked
    position q_xs is xbar_0 scaled by (move_chance_t - move_chance_s) /
    move_chance_t, a per-position *constant*, so it cannot reorder anything.

    Note this only has content under `mixture_sampling=marginal`. Under `edlm`
    q_xs at a masked position is a one-hot on the single drawn candidate's
    token, so every masked position scores the identical
    (move_chance_t - move_chance_s) / move_chance_t and the ranking degenerates
    to ties -- `xbar_conf` collapses into `random_k`. That is a real property of
    EDLM's sampler, not a limitation here, and it doubles as a check: the two
    arms should be indistinguishable when run under `edlm`.

    Caveat, stated plainly: once the position choice depends on model output the
    reverse kernel is no longer the forward posterior, so the importance-sampling
    identity that gives `w_n = softmax(lambda * r)` no longer holds. This is a
    decoder heuristic -- which is what MaskGIT-style confidence decoding already
    is -- not a sampler with the derivation behind it.
    """
    b, length, _ = q_xs.shape
    masked = (xt == self.mask_index)

    # The token to write if a position does unmask: q_xs conditioned on not
    # staying masked, i.e. renormalised over the real tokens.
    q_tok = q_xs.clone()
    q_tok[:, :, self.mask_index] = 0.0
    q_tok = q_tok / q_tok.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    tokens = _sample_categorical(q_tok)

    # How many to unmask: the Bernoulli sampler's *expected* count, so the noise
    # schedule is still respected and the last step (move_chance_s -> 0, so
    # p -> 1) necessarily empties the mask rather than leaving residue behind.
    p_unmask = ((move_chance_t - move_chance_s)
                / move_chance_t.clamp_min(1e-12)).reshape(b)
    n_masked = masked.sum(dim=1)
    k = torch.round(n_masked.to(p_unmask.dtype) * p_unmask).long()
    k = torch.minimum(torch.clamp(k, min=0), n_masked)

    if arm == 'random_k':
      score = torch.rand(b, length, device=q_xs.device, dtype=q_xs.dtype)
    elif arm in ('xtheta_conf', 'xbar_conf', 'blend_conf'):
      # xbar_0 and x_theta do not estimate the same thing with different noise:
      # as N grows xbar_0 converges to the *tilted* denoiser marginal and
      # x_theta is the *untilted* one, exactly. So blending them trades bias
      # (towards untilted) against the O(1/sqrt(N)) Monte Carlo error in
      # xbar_0 -- a shrinkage estimator with `conf_blend` as the shrinkage
      # weight, not a pure variance reduction. At lambda = 0 the two targets
      # coincide and the blend *is* pure variance reduction, which makes that
      # setting a clean implementation check.
      #
      # At a masked position q_xs is xbar_0 scaled by a per-position constant,
      # so renormalising over the real tokens recovers xbar_0 itself. Both
      # sources are put on that same footing before mixing.
      raw = getattr(self.config.guidance, 'conf_blend', 0.5)
      beta = {'xtheta_conf': 0.0, 'xbar_conf': 1.0}.get(
        arm, 0.5 if raw is None else float(raw))
      source = torch.zeros_like(q_xs)
      if beta < 1.0:
        source += (1.0 - beta) * _renorm_no_mask(x_theta, self.mask_index)
      if beta > 0.0:
        source += beta * _renorm_no_mask(q_xs, self.mask_index)
      # Max token probability -- the standard confidence used by MaskGIT-style
      # decoders. Entropy would be the other natural choice; max-prob keeps this
      # comparable to that literature.
      score = source.max(dim=-1).values
    elif arm == 'cv_conf':
      # Control-variate version of xbar_conf, and the principled one.
      #
      # xbar_uniform averages the SAME N candidates with uniform weights, so
      # E[xbar_uniform] = x_theta exactly. Its error is therefore a zero-mean
      # quantity that is *correlated* with the error in xbar_tilted, because
      # both are computed from the same draws. Subtracting it cancels the shared
      # noise without moving the target:
      #
      #     xbar_cv = xbar_tilted - c (xbar_uniform - x_theta)
      #
      # Unlike `blend_conf` this introduces **no bias** towards the untilted
      # marginal -- the correction has mean zero whatever c is -- and it costs no
      # extra reward evaluations, since the candidates were already scored.
      # Standard control-variate theory gives Var reduced by (1 - rho^2) at the
      # optimal c; c=1 is used here, which is optimal when the two estimators
      # carry the same-scale noise.
      if hist is None:
        raise NotImplementedError(
          "position_selection=cv_conf needs the candidate histograms, which "
          "are only built under mixture_sampling=marginal.")
      tilted, uniform = hist
      # `or 1.0` would be wrong here: 0.0 is falsy, so c=0 -- the setting that
      # turns the correction off and must reproduce `xbar_conf` exactly -- would
      # silently become c=1.
      raw_c = getattr(self.config.guidance, 'cv_coeff', 1.0)
      c = 1.0 if raw_c is None else float(raw_c)
      corrected = tilted - c * (uniform - x_theta.to(tilted.dtype))
      source = _renorm_no_mask(corrected.clamp_min(0.0), self.mask_index)
      score = source.max(dim=-1).values
    else:
      raise NotImplementedError(
        f"position_selection={arm} not implemented; expected 'bernoulli', "
        "'random_k', 'xtheta_conf', 'xbar_conf', 'blend_conf' or 'cv_conf'.")

    # Break ties at random. `argsort` is a stable sort, so equal scores come out
    # in index order and the top-k becomes "the leftmost positions" -- a
    # systematically different sampler, not the intended one. This matters most
    # under `edlm`, where every masked position scores identically and the whole
    # ranking is ties; without this the arm silently becomes left-to-right
    # decoding instead of degenerating to `random_k` as the theory says it must.
    score = score + torch.rand_like(score) * self._TIE_EPS
    # Already-unmasked positions must never be picked; -inf sorts them last, and
    # k <= n_masked keeps them out of the top-k regardless.
    score = score.masked_fill(~masked, float('-inf'))
    rank = score.argsort(dim=1, descending=True).argsort(dim=1)
    unmask = masked & (rank < k[:, None])
    return torch.where(unmask, tokens, xt)

  # --- sampling loop ---------------------------------------------------

  @torch.no_grad()
  def _diffusion_sample(
      self,
      classifier_model: typing.Optional[classifier.Classifier] = None,
      cond: typing.Optional[torch.Tensor] = None,
      eps: float = 1e-5,
  ):
    """Mirrors `Diffusion._diffusion_sample`, dispatching to `_our_denoise`.

    Any other `guidance.method` is delegated to the parent implementation so
    that baselines keep working through this subclass.
    """
    if getattr(self.config, 'guidance', None) is None or \
        self.config.guidance.method != 'ours':
      return super()._diffusion_sample(
        classifier_model=classifier_model, cond=cond, eps=eps)
    if self.parameterization == 'ar':
      raise NotImplementedError(
        'Reward-tilted posterior guidance is defined for diffusion '
        'posteriors and does not apply to AR models.')

    self._reward_cache.clear()
    self._valid_cache.clear()
    xt = self._sample_prior(
      self.config.sampling.batch_size,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, self.config.sampling.steps + 1, device=self.device)
    dt = (1 - eps) / self.config.sampling.steps
    pbar = tqdm(range(self.config.sampling.steps),
                desc='Sampling (ours)', leave=False)
    NFEs = 0
    cache = None

    for i in pbar:
      t = timesteps[i]
      if self.T > 0:  # t in {1/T,..., 1}, to match training
        t = (t * self.T).to(torch.int)
        t = t / self.T
        t += (1 / self.T)
      t_scalar = float(t)
      t = t * torch.ones(xt.shape[0], 1, device=self.device)
      if cache is None:
        NFEs += 1
      sigma_t, _ = self.noise(t)
      sigma_s, _ = self.noise(t - dt)
      if sigma_t.ndim > 1:
        sigma_t = sigma_t.squeeze(-1)
      if sigma_s.ndim > 1:
        sigma_s = sigma_s.squeeze(-1)
      assert sigma_t.ndim == 1, sigma_t.shape
      assert sigma_s.ndim == 1, sigma_s.shape
      move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
      move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]
      assert move_chance_t.ndim == 3, move_chance_t.shape

      xs, q_xs, cache = self._our_denoise(
        xt=xt,
        time_conditioning=sigma_t,
        move_chance_t=move_chance_t,
        move_chance_s=move_chance_s,
        t=t_scalar,
        cache=cache)
      pbar.set_postfix(
        NFEs=NFEs,
        ESS=round(self._last_ess, 2),
        vfrac=round(self._last_valid_frac, 3),
        allbad=round(self._last_all_invalid, 3),
        unif=round(self._last_uniform_frac, 3),
        pool=round(self._last_pool_valid_frac, 3),
        short=round(self._last_short_frac, 3),
        prob_check=(q_xs.sum() / xt.numel()).item(),
        nan_check=bool(q_xs.isnan().sum() > 0))
      if (not self.config.sampling.use_cache
          or not torch.allclose(xs, xt)):
        cache = None
      xt = xs
    return xt
