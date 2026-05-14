<div align="center">
  <img src="docs/flash/img/flash.png" alt="Realtime-VLA FLASH overview" width="100%"><br>
</div>


---

<div align="center">
  <a href="https://dexmal.github.io/realtime-vla-flash/"><b>Page</b></a> |
  <a href="https://arxiv.org/abs/2605.13778"><b>Paper</b></a> |
  <a href="https://huggingface.co/Dexmal/RealtimeVLA-Flash"><b>Model</b></a>
</div>


## News

- [2026/05] 🔥 Realtime-VLA FLASH code is now available.

## Highlights

Realtime-VLA FLASH is the first speculative inference framework for diffusion-based VLAs.

- Speculative inference as fast as 7.8 ms (2 views), enabling over 125 Hz real-time inference.
- VLM-aligned draft architecture with a deployment-friendly block design.
- FLASH serving with customized Triton kernels, achieving a 3.04× average task-level speedup.

## Installation

Follow [openpi README](README_OPENPI.md):

```bash
git clone --recurse-submodules https://github.com/dexmal/realtime-vla-flash
# Or if you already cloned the repo:
git submodule update --init --recursive
```

Install the Python environment with `uv`:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

LIBERO client/evaluation code can run in a separate environment. (see [LIBERO README](examples/libero/README.md)).

## Quick Start

First, convert the pretrained pi0 and draft checkpoints into the Triton weight layout.

```bash
uv run scripts/spec/triton/convert_for_triton.py \
   --mode base \
   --jax-path /path/to/jax/checkpoint \
   --output converted/base

uv run scripts/spec/triton/convert_for_triton.py \
   --mode draft \
   --draft-ckpt /path/to/draft_model.pt \
   --output converted/draft
```

Then start the policy server and the LIBERO client.

```bash
uv run scripts/spec/spec_serve_policy.py \
  --config pi0_libero \
  --base-triton-path converted/base \
  --draft-triton-path converted/draft \
  --task-suite-name libero_goal \
  --backend triton

uv run scripts/spec/spec_client_libero.py \
  --task-suite-name libero_goal
```

## Benchmark
You can check the inference time on your local machine by
```
uv run python scripts/spec/pi0_benchmark.py
```

## Train Draft Model

```bash
  uv run scripts/spec/enc_cache.py \
    --config pi0_libero \
    --checkpoint-dir /openpi-assets/checkpoints/pi0_libero_torch \
    --task-suite-name libero_goal \
    --output-dir /tmp/spec_quickstart_train/libero_goal_cache

  uv run scripts/spec/spec_draft_train.py \
    --cache-dir /tmp/spec_quickstart_train/libero_goal_cache \
    --output draft_model_goal_torch.pt
```

A typical workflow is:

1. Build a prefix-embedding cache with `scripts/spec/enc_cache.py`.
2. Train the draft head with `scripts/spec/spec_draft_train.py`.
3. Serve the FLASH policy with `scripts/spec/spec_serve_policy.py`.
4. Run LIBERO client evaluation or sweeps with `scripts/spec/spec_client_libero.py` or `scripts/spec/exp/run_sweep.py`.

## Citation

If you find this work useful, please cite the paper once the arXiv version is available:

```bibtex
@article{niu2026realtimevlaflash,
  title={Realtime-VLA FLASH: Speculative Inference Framework for Diffusion-based VLAs},
  author={Niu, Jiahui and Gu, Kefan and Zhao, Yucheng and Liang, Shengwen and Wang, Tiancai and Hu, Xing and Wang, Ying and Li, Huawei},
  journal={arXiv preprint arXiv:2605.13778},
  year={2026}
}
```

## Acknowledgements
- [dexmal/realtime-vla](https://github.com/dexmal/realtime-vla)
- [openpi](https://github.com/Physical-Intelligence/openpi)
