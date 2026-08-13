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


_WORKER = {}


def _init_reward_worker(reward_name: str, invalid_reward: float) -> None:
  rdkit.rdBase.DisableLog('rdApp.error')
  rdkit.rdBase.DisableLog('rdApp.warning')
  _WORKER['fn'] = REWARD_FNS[reward_name]
  _WORKER['invalid'] = invalid_reward


def _worker_reward(smiles: str) -> float:
  return smiles_reward(smiles, _WORKER['fn'], _WORKER['invalid'])


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
    # Effective sample size of the importance weights, averaged over the
    # batch, for the most recent step. ESS ~= 1 means the mixture collapsed
    # onto a single candidate and lambda_ is too large for this reward scale.
    self._last_ess: float = float('nan')
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
    return torch.tensor(
      [cache[s] for s in smiles_list],
      device=x0.device, dtype=torch.float32).view(n, b)

  def _mixture_weights(self, rewards: torch.Tensor) -> torch.Tensor:
    """w_n of Eq. (8); shape (N, B), summing to 1 over dim 0."""
    if self.config.guidance.normalize_reward:
      rewards = ((rewards - rewards.mean(dim=0, keepdim=True))
                 / (rewards.std(dim=0, keepdim=True) + 1e-6))
    return torch.softmax(
      float(self.config.guidance.lambda_) * rewards, dim=0)

  # --- posterior conditioned on a *given* x_0 -------------------------

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
    x0 = _sample_categorical(
      x_theta.unsqueeze(0).expand(n, b, length, vocab_size))

    # Eq. (8): w_n \propto exp(lambda * r(x_0^(n))).
    weights = self._mixture_weights(self._rewards(x0))
    # Effective sample size: N means the weights are uniform, 1 means they
    # collapsed onto a single candidate (lambda_ too large for this reward).
    self._last_ess = (1.0 / weights.pow(2).sum(dim=0)).mean().item()

    mixture_sampling = self.config.guidance.mixture_sampling
    if mixture_sampling == 'exact':
      # A mixture is sampled ancestrally: draw the component, then draw from
      # it. This keeps the within-sequence correlations of q(.|x_t, x_0^(n)).
      component = _sample_categorical(weights.t())  # (B,)
      x0_selected = torch.gather(
        x0, 0,
        component[None, :, None].expand(1, b, length)).squeeze(0)
      q_xs = self._posterior_given_x0(
        x0_selected, xt, move_chance_t, move_chance_s)
    elif mixture_sampling == 'marginal':
      # Average the N posteriors and sample positions independently. Matches
      # the mixture's per-position marginals but drops correlations.
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
    elif mixture_sampling == 'aggregate_x0':
      # The form used by discriminator guidance: collapse the N candidates
      # into a per-position weighted histogram xbar_0 and substitute *that*
      # for x_0 in the posterior, normalizing once at the end. Identical to
      # 'marginal' for absorbing-state diffusion (whose posterior normalizer
      # does not depend on x_0) and slightly different for uniform diffusion.
      # Note lambda_ = 0 makes xbar_0 an unbiased Monte Carlo estimate of
      # x_theta, so this mode reduces to unguided sampling as N grows.
      x0_one_hot = F.one_hot(x0, vocab_size).to(weights.dtype)
      x0_bar = (weights[:, :, None, None] * x0_one_hot).sum(dim=0)
      # xbar_0 is an atomic measure: a token that appears at position l in none
      # of the N candidates gets exactly zero mass, so the reverse step cannot
      # emit it. That makes the method an implicit stochastic truncation of
      # x_theta at roughly the 1/N tail. Mixing the denoiser marginal back in
      # with weight `support_floor` restores full support while keeping the
      # tilt; 0 reproduces the plain estimator.
      floor = float(getattr(self.config.guidance, 'support_floor', 0.0))
      if floor > 0.0:
        x0_bar = ((x0_bar + floor * x_theta.to(x0_bar.dtype))
                  / (1.0 + floor))
      q_xs = self._posterior_from_x0_dist(
        x0_bar, xt, move_chance_t, move_chance_s)
      q_xs = q_xs / q_xs.sum(dim=-1, keepdim=True)
    else:
      raise NotImplementedError(
        f"mixture_sampling={mixture_sampling} not implemented; expected "
        "'exact', 'marginal', or 'aggregate_x0'.")

    xs = _sample_categorical(q_xs)
    if self.diffusion == 'absorbing_state':
      copy_flag = (xt != self.mask_index).to(torch.bool)
      q_xs[copy_flag] = 0.0
      q_xs[copy_flag, xt[copy_flag]] = 1.0
      xs = torch.where(copy_flag, xt, xs)
    return xs, q_xs, {'log_x_theta': log_x_theta}

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
        prob_check=(q_xs.sum() / xt.numel()).item(),
        nan_check=bool(q_xs.isnan().sum() > 0))
      if (not self.config.sampling.use_cache
          or not torch.allclose(xs, xt)):
        cache = None
      xt = xs
    return xt
