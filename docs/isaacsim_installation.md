# Isaac Sim (Isaac Lab) installation

Play2Perfect runs in Isaac Sim via Isaac Lab. It requires **Python 3.11** (Isaac Sim 5.x / Isaac Lab 2.3.x requirement) and lives in a dedicated venv at `.venv_isaacsim/`.

## Prerequisites

- Python 3.11
- NVIDIA GPU with driver >= 525.60
- CUDA 12+
- `uv` for package management ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))

## Install

```bash
uv venv .venv_isaacsim --python 3.11

# PyTorch for CUDA 12.6 (Isaac Sim 5.x is built against this)
uv pip install --python .venv_isaacsim/bin/python torch --index-url https://download.pytorch.org/whl/cu126

# Vendored rl_games (PPO + SAPG)
uv pip install --python .venv_isaacsim/bin/python -e ./rl_games/

# Inference / tooling deps
uv pip install --python .venv_isaacsim/bin/python \
  omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy trimesh viser \
  requests tqdm tyro huggingface_hub "imageio[ffmpeg]" wandb termcolor

# Isaac Lab + Isaac Sim (~15 GB download; first launch builds RTX shaders, ~2-5 min)
uv pip install --python .venv_isaacsim/bin/python \
  "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

# typing_extensions fix: install AFTER isaaclab so it wins the resolution
# (tyro CLIs need NoExtraItems from typing_extensions>=4.13; isaaclab pulls 4.12.2).
uv pip install --python .venv_isaacsim/bin/python "typing_extensions>=4.13"

# Register repo-local packages (isaacsimenvs, evaluation, deployment)
uv pip install --python .venv_isaacsim/bin/python -e . --no-deps
```

`--no-deps` on the last line is required so the repo packages install cleanly without
re-resolving the heavy Isaac Sim stack.

Keep the version pins exact (`isaaclab==2.3.2.post1`, torch 2.7.x+cu126) — newer Isaac Lab
releases change `DirectRLEnv` / `UrdfConverter` APIs that `isaacsimenvs` depends on.

Verify:

```bash
.venv_isaacsim/bin/python -c "
import torch, isaaclab, isaacsimenvs
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('isaaclab:', isaaclab.__file__)
"
```

## Running

Non-interactive launches need the EULA env var:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Optional speedup — point the Omniverse shader cache at a local SSD instead of NFS:

```bash
export OMNI_KIT_CACHE_PATH=/scratch/$USER/ov_cache
mkdir -p "$OMNI_KIT_CACHE_PATH"
```

## Gotchas

- **AppLauncher before isaaclab imports.** Isaac Lab sub-namespaces (`isaaclab.sim`,
  `isaaclab.envs`, ...) only resolve after `AppLauncher(args)` runs. Any script importing
  `isaaclab.*` must instantiate `AppLauncher` first.
- **Kit shutdown can hang** after the work is done. Scripts flush stdout/stderr and call
  `os._exit(0)` rather than waiting for a clean Kit teardown.
- **One Isaac Sim instance per GPU.** Booting a second Kit process while another is running
  on the same GPU can crash the booting one mid-startup. Wait for training/eval/demo
  processes to finish before launching another.
