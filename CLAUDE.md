# 프로젝트 워크플로우

## 구조
- 코드는 **로컬에서 수정**하고, **GPU 서버에서 실행**한다
- 서버: `wp03052@143.248.84.179`
- 서버 프로젝트 경로: `/home/aailab/wp03052/Synthetic-Data/DLRT`
- 서버는 절대 직접 수정하지 않는다. 항상 로컬 수정 → 동기화 → 실행 순서
- 베이스 코드: https://github.com/kuleshov-group/discrete-diffusion-guidance

## 동기화
- 코드 수정 후 반드시 `bash sync.sh` 로 서버에 동기화한 뒤 실행할 것
- ⚠️ **반드시 프로젝트 루트에서 실행할 것.** sync.sh는 `./`를 소스로 쓰므로 다른 디렉토리에서
  실행하면 조용히 실패한다. (하위 디렉토리로 cd한 상태가 유지돼 실패한 적 있음)
- ⚠️ **서버에만 존재하는 폴더를 새로 만들면 즉시 sync.sh의 exclude에 추가할 것.**
  `rsync --delete`가 로컬에 없는 폴더를 삭제한다. `results/`를 빠뜨려서 실험 결과를
  전부 날린 적 있음. 현재 보호 대상: `outputs`, `watch_folder`, `results`,
  `.hf_cache`, `wandb`, `.data_cache`, `pdfs`, `*.ckpt`

## 서버 실행 방법
- Python 환경은 venv 사용. 위치: `/home/aailab/wp03052/venvs/dlrt_env`
  (2026-08-07에 `~/dlrt_env` → `~/venvs/` 로 이동. `bin/` 안 shebang과 `activate`의 `VIRTUAL_ENV`를
  치환했고, `flash_attn` / `mamba_ssm` / `causal_conv1d`의 컴파일된 CUDA 확장은 import 확인 완료.)
- 모든 서버 명령 앞에 activate 필요:
  `source /home/aailab/wp03052/venvs/dlrt_env/bin/activate`
- 실행 시 반드시 프로젝트 폴더로 cd 한 뒤, 환경변수를 프로젝트 내부로 지정할 것
  (HF 캐시가 홈 디렉토리 `~/.cache`에 생기는 것을 방지 — 작업 폴더 외부에 영향 금지):
  ```
  ssh wp03052@143.248.84.179 "source /home/aailab/wp03052/venvs/dlrt_env/bin/activate && \
    cd /home/aailab/wp03052/Synthetic-Data/DLRT && \
    export HF_HOME=\$PWD/.hf_cache && \
    export PYTHONPATH=\$PWD:\$PWD/guidance_eval:\$HF_HOME/modules && \
    python main.py ..."
  ```
  (원본 repo의 `setup_env.sh`가 하는 일과 동일. 단 setup_env.sh는 conda용이므로 직접 쓰지 않는다)

## 의존성 (2026-08-01 설치 완료)
- 서버 venv는 python 3.10.13 (`/usr/local/bin/python3.10`) 기반. 원본 스펙은 3.9지만 3.10으로 정상 동작 확인
- `requirements.yaml`은 원본 repo의 conda 스펙(참고용). 실제 설치는 우리가 만든 `requirements.txt` 사용
- `setuptools<81` 필수 — lightning 2.2.1이 pkg_resources를 요구함 (81+에서 제거)
- `causal-conv1d`, `mamba-ssm`, `flash-attn`은 **torch 설치 후** `--no-build-isolation`으로 별도 설치
- flash-attn은 setup.py의 cross-device rename 버그로 pip 직접 설치가 실패함.
  GitHub releases에서 휠을 직접 받아 설치할 것:
  `flash_attn-2.7.2.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
- 서버 GPU: RTX 3090 × 4 (24GB, sm_86), driver CUDA 12.9, torch는 cu121 휠

## 서버에서 자주 밟은 함정
- `loader.num_workers` 기본값이 CPU 수(40). forked worker가 CUDA를 건드려
  `CUDA error: initialization error`로 죽는다 → `loader.num_workers=0
  loader.persistent_workers=False`
- hydra override: 이미 config에 있는 키에 `+`를 붙이면 실패. 새 키는 `+`,
  덮어쓰기는 접두어 없이, 둘 다 되게 하려면 `++`
- hydra는 실행 시 작업 디렉토리를 `hydra.run.dir`로 바꾼다. 스크립트에 넘기는
  출력 경로는 **반드시 절대경로**로 줄 것 (상대경로면 샘플링 다 끝낸 뒤 저장에서 죽는다)
- wandb 계정 없음 → `WANDB_MODE=offline WANDB_DIR=$PWD`

## 규칙
- 디버깅/테스트 목적의 짧은 실행만 SSH로 직접 돌릴 것
- 긴 학습/스윕은 반드시 서버 `tmux` 세션으로 띄울 것. SSH에 붙여 돌리면 연결이 끊기면 죽는다.
  현재 사용 중인 세션: `cbg`(classifier 학습), `cmp`(비교 스윕), `agg`(표 자동 생성)
- 서버에서는 프로젝트 폴더(`/home/aailab/wp03052/Synthetic-Data/DLRT`)와 venv 폴더(`/home/aailab/wp03052/venvs/dlrt_env`) 외의 어떤 경로도 생성/수정하지 말 것
- 의존성을 변경했으면 서버 venv에도 동일하게 pip install 할 것
