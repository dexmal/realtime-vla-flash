import dataclasses
import json
import os
from bisect import bisect_right
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
import tyro
from safetensors import safe_open as _safe_open
from safetensors.torch import load_file as _load_safetensors

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from openpi.models_pytorch.draft import DraftChunkHead

_SLIDING_CACHE_SAMPLE_SEMANTICS = "sliding_chunk_shift"


def _setup_ddp() -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if use_ddp and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", device_id=device if device.type == "cuda" else None)

    return use_ddp, rank, world_size, local_rank, device


def _dist_barrier(device: torch.device) -> None:
    if device.type == "cuda":
        dist.barrier(device_ids=[int(device.index if device.index is not None else 0)])
    else:
        dist.barrier()


_MASK64 = 0xFFFFFFFFFFFFFFFF


def _splitmix64_np(x: np.ndarray) -> np.ndarray:
    z = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(_MASK64)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(_MASK64)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(_MASK64)
    return (z ^ (z >> np.uint64(31))) & np.uint64(_MASK64)


def _episode_val_mask_np(episode_index: np.ndarray, *, val_frac: float, split_seed: int) -> np.ndarray:
    frac = float(val_frac)
    if frac <= 0.0:
        return np.zeros_like(episode_index, dtype=np.bool_)
    if frac >= 1.0:
        return np.ones_like(episode_index, dtype=np.bool_)
    ep_u = np.asarray(episode_index, dtype=np.uint64)
    key = (ep_u + np.uint64(int(split_seed)) * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(_MASK64)
    h = _splitmix64_np(key)
    # Float conversion is fine here; we only need a deterministic partition.
    return (h.astype(np.float64) / float(1 << 64)) < frac


def _weighted_huber_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    beta: float,
    step_weights: torch.Tensor,
) -> torch.Tensor:
    if pred.shape != tgt.shape:
        raise ValueError(f"shape mismatch pred={tuple(pred.shape)} tgt={tuple(tgt.shape)}")
    if pred.ndim != 3:
        raise ValueError(f"expected (B,M,D), got pred.shape={tuple(pred.shape)}")
    m = int(pred.shape[1])
    if m <= 0:
        return torch.zeros((), device=pred.device, dtype=torch.float32)
    if step_weights.ndim != 1 or int(step_weights.shape[0]) != m:
        raise ValueError(f"expected step_weights to be (M,) with M={m}, got shape={tuple(step_weights.shape)}")

    w = step_weights.to(device=pred.device, dtype=torch.float32)
    w = w / w.sum().clamp_min(1e-8)
    per_elem = F.smooth_l1_loss(pred, tgt, reduction="none", beta=float(beta))
    per_t = per_elem.mean(dim=2)
    per_sample = (per_t * w[None, :]).sum(dim=1)
    return per_sample.mean()


def _time_weights(*, length: int, gamma: float, device: torch.device | str) -> torch.Tensor:
    m = int(length)
    if m <= 0:
        return torch.zeros((0,), device=device, dtype=torch.float32)
    w = torch.pow(torch.as_tensor(float(gamma), device=device, dtype=torch.float32), torch.arange(m, device=device))
    return w / (w.sum() + 1e-8)


def _sampled_prefix_len(
    *,
    length: int,
    prefix_cap: int,
    generator: torch.Generator,
    device: torch.device | str,
) -> int:
    m = int(length)
    if m <= 0:
        return 0
    cap = int(prefix_cap) if int(prefix_cap) > 0 else m
    cap = int(max(1, min(m, cap)))
    sampled = torch.randint(1, cap + 1, (1,), device=device, generator=generator)
    return int(sampled.item())


def _sampled_prefix_step_weights(
    *,
    length: int,
    prefix_len: int,
    gamma_prefix: float,
    gamma_tail: float,
    tail_weight: float,
    device: torch.device | str,
) -> torch.Tensor:
    m = int(length)
    if m <= 0:
        return torch.zeros((0,), device=device, dtype=torch.float32)

    prefix_len = int(max(1, min(m, int(prefix_len))))
    prefix = torch.pow(
        torch.as_tensor(float(gamma_prefix), device=device, dtype=torch.float32),
        torch.arange(prefix_len, device=device),
    )
    if prefix_len >= m:
        return prefix / prefix.sum().clamp_min(1e-8)

    tail_len = int(m - prefix_len)
    tail = torch.pow(
        torch.as_tensor(float(gamma_tail), device=device, dtype=torch.float32),
        torch.arange(tail_len, device=device),
    ) * float(tail_weight)
    weights = torch.cat([prefix, tail], dim=0)
    return weights / weights.sum().clamp_min(1e-8)


def _loss_step_weights(
    *,
    length: int,
    mode: str,
    gamma: float,
    prefix_cap: int,
    gamma_prefix: float,
    gamma_tail: float,
    tail_weight: float,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    eval_prefix_len: int | None = None,
) -> torch.Tensor:
    if str(mode) == "full_chunk_gamma":
        return _time_weights(length=int(length), gamma=float(gamma), device=device)
    if str(mode) != "sampled_prefix":
        raise ValueError(f"unsupported loss_prefix_mode={mode!r}")

    if eval_prefix_len is not None:
        prefix_len = int(eval_prefix_len)
    else:
        if generator is None:
            raise ValueError("generator is required for sampled_prefix training weights")
        prefix_len = _sampled_prefix_len(
            length=int(length),
            prefix_cap=int(prefix_cap),
            generator=generator,
            device=device,
        )
    return _sampled_prefix_step_weights(
        length=int(length),
        prefix_len=int(prefix_len),
        gamma_prefix=float(gamma_prefix),
        gamma_tail=float(gamma_tail),
        tail_weight=float(tail_weight),
        device=device,
    )


def _manifest_target_source(manifest: dict[str, Any]) -> str:
    run_spec = manifest.get("run_spec", {})
    source = manifest.get("target_source", None)
    if source is None and isinstance(run_spec, dict):
        source = run_spec.get("target_source", None)
    return str(source or "gt")


def _manifest_teacher_noise_mode(manifest: dict[str, Any]) -> str:
    run_spec = manifest.get("run_spec", {})
    noise_mode = manifest.get("teacher_noise_mode", None)
    if noise_mode is None and isinstance(run_spec, dict):
        noise_mode = run_spec.get("teacher_noise_mode", None)
    return str(noise_mode or "none")


def _manifest_sample_semantics(manifest: dict[str, Any]) -> str:
    value = manifest.get("sample_semantics", None)
    if value is None:
        run_spec = manifest.get("run_spec", {})
        if isinstance(run_spec, dict):
            value = run_spec.get("sample_semantics", None)
    if value is None:
        raise ValueError("cache manifest missing required field sample_semantics; rebuild cache with enc_cache.py")
    return str(value)


def _require_supported_cache_semantics(ds: "_ShardCacheDataset") -> None:
    if str(ds.sample_semantics) != _SLIDING_CACHE_SAMPLE_SEMANTICS:
        raise ValueError(
            f"cache sample_semantics={ds.sample_semantics!r} is unsupported; rebuild cache with "
            f"sample_semantics={_SLIDING_CACHE_SAMPLE_SEMANTICS!r}"
        )


def _build_draft_head(
    *,
    img_dim: int,
    chunk_m: int,
    out_dim: int,
    device: torch.device | str,
    state_dict: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> torch.nn.Module:
    meta = dict(meta or {})
    state_dict = state_dict or {}
    gate_proj_weight = state_dict.get("_gemma_block.mlp.gate_proj.weight")
    if gate_proj_weight is None:
        gate_proj_weight = state_dict.get("mlp.gate_proj.weight")
    hidden_dim = int(gate_proj_weight.shape[0]) if gate_proj_weight is not None else 256
    num_heads = int(meta.get("draft_num_heads", DraftChunkHead._resolve_num_heads(int(img_dim))))
    num_kv_heads = int(meta.get("draft_num_kv_heads", 1))
    head_dim = int(meta.get("draft_head_dim", max(1, int(img_dim) // max(1, num_heads))))
    return DraftChunkHead(
        img_dim=int(img_dim),
        chunk_m=int(chunk_m),
        hidden_dim=int(hidden_dim),
        out_dim=int(out_dim),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
    ).to(device)


def _head_uses_seed_actions(head: torch.nn.Module) -> bool:
    del head
    return False


def _compute_draft_training_loss(
    *,
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    gamma: float,
    loss_prefix_mode: str,
    gamma_prefix: float,
    gamma_tail: float,
    tail_weight: float,
    beta: float,
    prefix_cap: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    eval_prefix_len: int | None = None,
    head: torch.nn.Module | None = None,
    gripper_aux: torch.Tensor | None = None,
    gripper_mode: str | None = None,
    gripper_loss_weight: float = 1.0,
    gripper_delta_weight: float = 0.0,
    gripper_transition_threshold: float = 0.0,
    gripper_transition_weight: float = 1.0,
) -> torch.Tensor:
    del head, gripper_aux, gripper_mode, gripper_loss_weight, gripper_delta_weight, gripper_transition_threshold
    del gripper_transition_weight
    loss_weights = _loss_step_weights(
        length=int(pred_actions.shape[1]),
        mode=str(loss_prefix_mode),
        gamma=float(gamma),
        prefix_cap=int(prefix_cap),
        gamma_prefix=float(gamma_prefix),
        gamma_tail=float(gamma_tail),
        tail_weight=float(tail_weight),
        device=device,
        generator=generator,
        eval_prefix_len=eval_prefix_len,
    )
    return _draft_loss(
        pred_actions=pred_actions,
        target_actions=target_actions,
        beta=float(beta),
        step_weights=loss_weights,
    )


def _run_draft_head(
    head: torch.nn.Module,
    *,
    primary_input: torch.Tensor | None = None,
    secondary_input: torch.Tensor | None = None,
    tertiary_input: torch.Tensor | None = None,
    img_front: torch.Tensor | None = None,
    img_wrist: torch.Tensor | None = None,
    robot_state: torch.Tensor,
    last_actions: torch.Tensor,
    seed_actions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    del img_front, img_wrist, seed_actions
    ddp_head = isinstance(head, torch.nn.parallel.DistributedDataParallel)
    module = head.module if ddp_head else head
    if not isinstance(module, DraftChunkHead):
        raise TypeError(f"unsupported draft head type: {type(module)!r}")
    if primary_input is None or secondary_input is None:
        raise ValueError("draft head inputs are missing")
    if tertiary_input is None:
        raise ValueError("draft head requires tertiary_input=prefix_att_masks")
    caller = head if ddp_head else module
    pred = caller(
        prefix_embs=primary_input,
        prefix_pad_masks=secondary_input,
        prefix_att_masks=tertiary_input,
        robot_state=robot_state,
        last_actions=last_actions,
    ).to(dtype=torch.float32)
    return pred, None


def _draft_head_meta(head: torch.nn.Module) -> dict[str, Any]:
    module = head.module if isinstance(head, torch.nn.parallel.DistributedDataParallel) else head
    if not isinstance(module, DraftChunkHead):
        raise TypeError(f"unsupported draft head type: {type(module)!r}")
    return {
        "draft_arch": "vlm_block",
        "draft_input_mode": "prefix_embs",
        "draft_num_heads": int(module.num_heads),
        "draft_num_kv_heads": int(module.num_kv_heads),
        "draft_head_dim": int(module.head_dim),
        "use_last_actions": False,
        "use_seed_actions": False,
    }


def _extract_draft_ckpt_meta_state_dict(ckpt: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    meta: dict[str, Any] = {}
    if isinstance(ckpt, dict) and "draft_head" in ckpt:
        sd = dict(ckpt.get("draft_head", {}) or {})
        meta = dict(ckpt.get("meta", {}) or {})
        return meta, sd
    if isinstance(ckpt, dict):
        return {}, dict(ckpt)
    raise ValueError("draft checkpoint must be a state_dict or a dict with key `draft_head`")


def _select_build_state_dict_for_draft_head(
    resume_state_dict: dict[str, Any],
    cache_init: tuple[dict[str, Any], dict[str, Any]] | None,
) -> dict[str, Any]:
    if resume_state_dict:
        return resume_state_dict
    if cache_init is not None:
        return cache_init[1]
    return {}


def _require_cache_compatible_with_head(ds: "_ShardCacheDataset") -> None:
    if str(getattr(ds, "draft_input_mode", "prefix_embs")) != "prefix_embs":
        raise ValueError("draft head requires a prefix-embedding cache; rebuild cache with enc_cache.py")


def _load_cache_init(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    path = Path(run_dir) / "draft_vlm_block_init.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"invalid draft init checkpoint at {path}")
    meta = dict(ckpt.get("meta", {}) or {})
    state_dict = dict(ckpt.get("gemma_block", {}) or {})
    if not state_dict:
        raise ValueError(f"draft init checkpoint missing gemma_block weights at {path}")
    return meta, state_dict


def _draft_loss(
    *,
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    beta: float,
    step_weights: torch.Tensor,
) -> torch.Tensor:
    """Step-weighted Huber loss across all action dimensions."""
    d = int(min(int(pred_actions.shape[2]), int(target_actions.shape[2])))
    if d <= 0:
        return torch.zeros((), device=pred_actions.device, dtype=torch.float32)
    return _weighted_huber_loss(
        pred_actions[:, :, :d],
        target_actions[:, :, :d],
        beta=float(beta),
        step_weights=step_weights,
    )


def _stepwise_pose_rms(pred_actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
    d = int(min(int(pred_actions.shape[2]), int(target_actions.shape[2]), 6))
    if d <= 0:
        return torch.zeros((int(pred_actions.shape[1]),), device=pred_actions.device, dtype=torch.float32)
    diff = (pred_actions[:, :, :d] - target_actions[:, :, :d]).to(dtype=torch.float32)
    return torch.sqrt(torch.mean(diff * diff, dim=(0, 2)))


def _print_diagnostic_sample(
    *,
    tag: str,
    target_source: str,
    sample_idx: int,
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    print_steps: int,
) -> None:
    steps = int(min(print_steps, pred_actions.shape[1], target_actions.shape[1]))
    pose_rms = _stepwise_pose_rms(pred_actions[:, :steps, :], target_actions[:, :steps, :]).detach().cpu().numpy()
    pred_grip = pred_actions[0, :steps, 6].detach().cpu().numpy() if int(pred_actions.shape[2]) >= 7 else np.zeros((steps,))
    teacher_grip = (
        target_actions[0, :steps, 6].detach().cpu().numpy() if int(target_actions.shape[2]) >= 7 else np.zeros((steps,))
    )
    pred_grip_delta = np.diff(pred_grip) if steps > 1 else np.zeros((0,), dtype=np.float32)
    teacher_grip_delta = np.diff(teacher_grip) if steps > 1 else np.zeros((0,), dtype=np.float32)
    print(
        f"diagnose[{tag}] sample={sample_idx} target_source={target_source} "
        f"pose_rms_step={np.array2string(pose_rms, precision=4, separator=', ')}"
    )
    print(f"diagnose[{tag}] sample={sample_idx} pred_grip={np.array2string(pred_grip, precision=4, separator=', ')}")
    print(f"diagnose[{tag}] sample={sample_idx} teacher_grip={np.array2string(teacher_grip, precision=4, separator=', ')}")
    print(f"diagnose[{tag}] sample={sample_idx} pred_grip_delta={np.array2string(pred_grip_delta, precision=4, separator=', ')}")
    print(
        f"diagnose[{tag}] sample={sample_idx} "
        f"teacher_grip_delta={np.array2string(teacher_grip_delta, precision=4, separator=', ')}"
    )


class _ShardCacheDataset(torch.utils.data.Dataset):
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        manifest = json.loads((self._run_dir / "manifest.json").read_text())
        self.target_source = _manifest_target_source(manifest)
        self.teacher_noise_mode = _manifest_teacher_noise_mode(manifest)
        self.sample_semantics = _manifest_sample_semantics(manifest)
        self.draft_input_mode = str(manifest.get("draft_input_mode", "prefix_embs"))
        self.feature_dim = int(manifest.get("feature_dim", 0))
        self.robot_state_dim = int(manifest.get("robot_state_dim", 0))
        self.action_dim = int(manifest.get("action_dim", 0))
        self.chunk_m = int(manifest.get("chunk_m", 0))
        self.out_dim = int(manifest.get("out_dim", 7))
        self.draft_history_len = int(manifest.get("draft_history_len", 0))

        shards = list(manifest["shards"])
        if not shards:
            raise ValueError(f"no shards in manifest: {self._run_dir / 'manifest.json'}")
        self._shards: list[dict[str, Any]] = shards

        offsets = [0]
        total = 0
        for s in self._shards:
            n = int(s["num_samples"])
            total += n
            offsets.append(total)
        self._offsets = offsets
        self._total = int(total)

        self._loaded_shard_id: int | None = None
        self._loaded: dict[str, torch.Tensor] | None = None

        if self.feature_dim <= 0 or self.robot_state_dim <= 0 or self.action_dim <= 0 or self.chunk_m <= 0:
            first_path = self._run_dir / str(self._shards[0]["path"])
            tensors = _load_safetensors(str(first_path))
            if "prefix_embs" not in tensors:
                raise ValueError("prefix cache missing `prefix_embs`; rebuild cache with enc_cache.py")
            self.feature_dim = int(tensors["prefix_embs"].shape[2])
            self.robot_state_dim = int(tensors["robot_state"].shape[1])
            self.action_dim = int(tensors["last_actions"].shape[2])
            self.draft_history_len = int(tensors["last_actions"].shape[1])
            self.chunk_m = int(tensors["targets"].shape[1])
            self.out_dim = int(tensors["targets"].shape[2])

    def __len__(self) -> int:
        return int(self._total)

    def _load_shard(self, shard_id: int) -> None:
        s = self._shards[int(shard_id)]
        path = self._run_dir / str(s["path"])
        self._loaded = _load_safetensors(str(path))
        self._loaded_shard_id = int(shard_id)

    def __getitem__(self, idx: int):
        i = int(idx)
        if i < 0 or i >= self._total:
            raise IndexError(idx)
        shard_id = bisect_right(self._offsets, i) - 1
        offset = i - int(self._offsets[shard_id])
        if self._loaded_shard_id != int(shard_id) or self._loaded is None:
            self._load_shard(int(shard_id))
        assert self._loaded is not None

        try:
            primary = self._loaded["prefix_embs"][offset]
            secondary = self._loaded["prefix_pad_masks"][offset]
            tertiary = self._loaded["prefix_att_masks"][offset]
        except KeyError as exc:
            raise ValueError("prefix cache missing required prefix fields; rebuild cache with enc_cache.py") from exc
        robot = self._loaded["robot_state"][offset]
        last = self._loaded["last_actions"][offset]
        tgt = self._loaded["targets"][offset]
        return primary, secondary, tertiary, robot, last, tgt


@dataclasses.dataclass
class Args:
    mode: Literal["train", "eval"] = "train"

    device: str | None = None

    max_samples: int | None = None

    # Draft head + training
    gripper_mode: Literal["continuous", "binary"] = "continuous"
    batch_size: int = 64
    epochs: int = 500
    lr: float = 2e-3
    seed: int = 0
    out_ckpt: str = "data/spec_metrics/draft_chunk_head.pt"
    ckpt: str | None = None

    # Eval + splitting
    val_frac: float = 0.1
    split_seed: int | None = None
    eval_split: Literal["val", "train", "all"] = "val"
    eval_interval_epochs: int = 20
    eval_max_samples: int | None = 2_000
    eval_exec_steps: int = 12
    eval_seed_mode: Literal["gt", "zero", "copy_last"] = "gt"
    diagnose_samples: int = 0
    diagnose_print_steps: int = 8
    diagnose_history_probe: bool = False

    # Loss / early stop
    loss_gamma: float = 0.95
    loss_prefix_mode: Literal["sampled_prefix", "full_chunk_gamma"] = "sampled_prefix"
    loss_prefix_cap: int = 16
    loss_gamma_prefix: float = 0.9
    loss_gamma_tail: float = 1.0
    loss_tail_weight: float = 0.1
    huber_beta: float = 1.0
    early_stop_patience_evals: int = 3
    early_stop_min_delta: float = 0.0

    # Kept for CLI compatibility; the current VLM-block draft head does not use seed-action augmentation.
    seed_zero_p: float = 0.1
    seed_noise_p: float = 0.1
    seed_copy_prev_p: float = 0.1
    seed_noise_std: float = 0.02

    # Cache mode (skip image decode + encoder)
    cache_run_dir: str | None = None
    resume_from_ckpt: str | None = None
    num_workers: int = 4


def _resolve_ckpt_path(args: Args) -> Path:
    return Path(args.ckpt or args.out_ckpt)


def _cache_build_split_indices(
    ds: _ShardCacheDataset,
    *,
    max_samples: int | None,
    val_frac: float,
    split_seed: int,
) -> tuple[list[int], list[int], int, int, int]:
    n_total = int(len(ds))
    n_use = int(n_total if max_samples is None else min(n_total, int(max_samples)))

    train_indices: list[int] = []
    val_indices: list[int] = []
    train_eps: set[int] = set()
    val_eps: set[int] = set()

    for shard_id, s in enumerate(ds._shards):  # noqa: SLF001
        shard_offset = int(ds._offsets[int(shard_id)])  # noqa: SLF001
        if shard_offset >= n_use:
            break
        shard_path = ds._run_dir / str(s["path"])  # noqa: SLF001
        with _safe_open(str(shard_path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if "episode_index" in keys:
                ep_t = f.get_tensor("episode_index")
            elif "dataset_index" in keys:
                ep_t = f.get_tensor("dataset_index")
            else:
                raise ValueError(f"cache shard missing episode_index/dataset_index: {shard_path}")

        ep = ep_t.to(dtype=torch.int64).cpu().numpy()
        shard_take = int(min(int(ep.shape[0]), n_use - shard_offset))
        if shard_take <= 0:
            continue
        ep = ep[:shard_take]

        mask_val = _episode_val_mask_np(ep, val_frac=val_frac, split_seed=split_seed)
        local = np.arange(shard_take, dtype=np.int64) + np.int64(shard_offset)
        val_indices.extend(local[mask_val].tolist())
        train_indices.extend(local[~mask_val].tolist())

        if ep.size:
            if mask_val.any():
                val_eps.update(np.unique(ep[mask_val]).astype(np.int64).tolist())
            if (~mask_val).any():
                train_eps.update(np.unique(ep[~mask_val]).astype(np.int64).tolist())

    return train_indices, val_indices, len(train_eps), len(val_eps), n_use


@torch.inference_mode()
def _eval_head_from_cache(
    head: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    target_source: str,
    loss_prefix_mode: str,
    gamma: float,
    gamma_prefix: float,
    gamma_tail: float,
    tail_weight: float,
    beta: float,
    exec_steps: int,
    max_samples: int | None,
    eval_seed_mode: str = "gt",
    diagnose_samples: int = 0,
    diagnose_print_steps: int = 8,
    diagnose_history_probe: bool = False,
) -> dict[str, float]:
    head.eval()

    sum_huber = 0.0
    sum_exec = 0.0
    sum_all = 0.0
    sum_pos = 0.0
    sum_rot = 0.0
    sum_grip = 0.0
    count = 0
    diagnose_done = 0

    out_dim = int(getattr(head, "out_dim", 7))
    for primary, secondary, tertiary, robot, last, tgt in loader:
        if max_samples is not None and count >= int(max_samples):
            break
        primary = primary.to(device=device, non_blocking=True)
        secondary = secondary.to(device=device, non_blocking=True)
        tertiary = tertiary.to(device=device, non_blocking=True)
        robot = robot.to(device=device, non_blocking=True)
        last = last.to(device=device, non_blocking=True)
        tgt = tgt.to(device=device, dtype=torch.float32, non_blocking=True)
        if str(eval_seed_mode) not in {"gt", "zero", "copy_last"}:
            raise ValueError(f"unsupported eval_seed_mode={eval_seed_mode!r}")

        b = int(primary.shape[0])
        if max_samples is not None and (count + b) > int(max_samples):
            take = int(max_samples) - int(count)
            primary = primary[:take]
            secondary = secondary[:take]
            tertiary = tertiary[:take]
            robot = robot[:take]
            last = last[:take]
            tgt = tgt[:take]
            b = int(take)

        pred, _gripper_aux = _run_draft_head(
            head,
            primary_input=primary,
            secondary_input=secondary,
            tertiary_input=tertiary,
            robot_state=robot,
            last_actions=last,
        )
        pred = pred.to(dtype=torch.float32)

        m = int(min(int(pred.shape[1]), int(tgt.shape[1])))
        d = int(min(int(out_dim), int(pred.shape[2]), int(tgt.shape[2])))
        if m <= 0 or d <= 0:
            continue
        pred_m = pred[:, :m, :d]
        tgt_m = tgt[:, :m, :d]
        loss = _compute_draft_training_loss(
            pred_actions=pred_m,
            target_actions=tgt_m,
            gamma=float(gamma),
            loss_prefix_mode=str(loss_prefix_mode),
            gamma_prefix=float(gamma_prefix),
            gamma_tail=float(gamma_tail),
            tail_weight=float(tail_weight),
            beta=float(beta),
            prefix_cap=int(exec_steps),
            device=pred_m.device,
            eval_prefix_len=int(min(int(m), max(1, int(exec_steps)))),
        )
        sum_huber += float(loss.item()) * float(b)

        diff = (pred_m - tgt_m).to(dtype=torch.float32)
        exec_h = int(min(int(m), max(1, int(exec_steps))))
        rms_exec = torch.sqrt(torch.mean(diff[:, :exec_h, :] * diff[:, :exec_h, :], dim=(1, 2)))
        rms_all = torch.sqrt(torch.mean(diff * diff, dim=(1, 2)))
        rms_pos = torch.sqrt(torch.mean(diff[:, :, 0:3] * diff[:, :, 0:3], dim=(1, 2))) if d >= 3 else rms_all * float("nan")
        rms_rot = (
            torch.sqrt(torch.mean(diff[:, :, 3:6] * diff[:, :, 3:6], dim=(1, 2))) if d >= 6 else rms_all * float("nan")
        )
        rms_grip = (
            torch.sqrt(torch.mean(diff[:, :, 6:7] * diff[:, :, 6:7], dim=(1, 2))) if d >= 7 else rms_all * float("nan")
        )

        sum_exec += float(rms_exec.sum().item())
        sum_all += float(rms_all.sum().item())
        sum_pos += float(rms_pos.sum().item()) if d >= 3 else 0.0
        sum_rot += float(rms_rot.sum().item()) if d >= 6 else 0.0
        sum_grip += float(rms_grip.sum().item()) if d >= 7 else 0.0
        count += b

        if diagnose_done < int(max(0, int(diagnose_samples))):
            _print_diagnostic_sample(
                tag=f"cache:{eval_seed_mode}",
                target_source=target_source,
                sample_idx=diagnose_done,
                pred_actions=pred_m[:1],
                target_actions=tgt_m[:1],
                print_steps=int(diagnose_print_steps),
            )
            diagnose_done += 1

    denom = float(count) if count > 0 else float("nan")
    return {
        "samples": float(count),
        "weighted_huber": (sum_huber / denom) if count > 0 else float("nan"),
        "rms_exec": (sum_exec / denom) if count > 0 else float("nan"),
        "rms_all": (sum_all / denom) if count > 0 else float("nan"),
        "rms_pos": (sum_pos / denom) if count > 0 else float("nan"),
        "rms_rot": (sum_rot / denom) if count > 0 else float("nan"),
        "rms_grip": (sum_grip / denom) if count > 0 else float("nan"),
    }


def _train_from_cache(args: Args) -> None:
    use_ddp, rank, world_size, _local_rank, device = _setup_ddp()
    torch.manual_seed(int(args.seed) + int(rank))
    np.random.seed(int(args.seed) + int(rank))

    if not use_ddp:
        device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)

    ds = _ShardCacheDataset(Path(args.cache_run_dir or ""))
    _require_supported_cache_semantics(ds)
    _require_cache_compatible_with_head(ds)
    split_seed = int(args.seed if args.split_seed is None else args.split_seed)
    if (not use_ddp) or rank == 0:
        print(
            f"cache_target_source={ds.target_source} teacher_noise_mode={ds.teacher_noise_mode} "
            f"sample_semantics={ds.sample_semantics}"
        )

    train_idx, val_idx, n_train_eps, n_val_eps, n_use = _cache_build_split_indices(
        ds,
        max_samples=args.max_samples,
        val_frac=float(args.val_frac),
        split_seed=split_seed,
    )
    if (not use_ddp) or rank == 0:
        print(
            f"cache_samples_total={n_use} train_samples={len(train_idx)} val_samples={len(val_idx)} "
            f"train_episodes={n_train_eps} val_episodes={n_val_eps} val_frac={float(args.val_frac):.3f} split_seed={split_seed}"
        )

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    resume_ckpt: dict[str, Any] | None = None
    if args.resume_from_ckpt is not None:
        resume_ckpt = torch.load(args.resume_from_ckpt, map_location="cpu")
    resume_meta, resume_sd = _extract_draft_ckpt_meta_state_dict(resume_ckpt or {})
    cache_init = None if args.resume_from_ckpt is not None else _load_cache_init(Path(args.cache_run_dir or ""))
    build_meta = dict(cache_init[0]) if cache_init is not None else {}
    build_meta.update(resume_meta)
    build_sd = _select_build_state_dict_for_draft_head(resume_sd, cache_init)
    head = _build_draft_head(
        img_dim=int(ds.feature_dim),
        chunk_m=int(ds.chunk_m),
        out_dim=int(ds.out_dim),
        device=device,
        state_dict=build_sd,
        meta=build_meta,
    )
    if cache_init is not None and not resume_sd:
        missing, unexpected = head._gemma_block.load_state_dict(cache_init[1], strict=False)  # noqa: SLF001
        if (not use_ddp) or rank == 0:
            print(f"loaded_vlm_block_init={Path(args.cache_run_dir or '') / 'draft_vlm_block_init.pt'} missing={len(missing)} unexpected={len(unexpected)}")
    if args.resume_from_ckpt is not None:
        head.load_state_dict(resume_sd, strict=True)

    if use_ddp:
        head = torch.nn.parallel.DistributedDataParallel(
            head,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    train_sampler = None
    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(args.num_workers),
        persistent_workers=int(args.num_workers) > 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    optim = torch.optim.AdamW(
        (head.module if isinstance(head, torch.nn.parallel.DistributedDataParallel) else head).parameters(),
        lr=float(args.lr),
    )

    best_val_rms_exec = float("inf")
    best_val_huber = float("inf")
    bad_evals = 0

    for epoch in range(int(args.epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        head.train()
        g = torch.Generator(device=device)
        g.manual_seed(int(args.seed) + 1000 * int(epoch) + int(rank))

        sum_loss = 0.0
        sum_all = 0.0
        count = 0

        for primary, secondary, tertiary, robot, last, tgt in train_loader:
            primary = primary.to(device=device, non_blocking=True)
            secondary = secondary.to(device=device, non_blocking=True)
            tertiary = tertiary.to(device=device, non_blocking=True)
            robot = robot.to(device=device, non_blocking=True)
            last = last.to(device=device, non_blocking=True)
            tgt = tgt.to(device=device, dtype=torch.float32, non_blocking=True)

            pred, _gripper_aux = _run_draft_head(
                head,
                primary_input=primary,
                secondary_input=secondary,
                tertiary_input=tertiary,
                robot_state=robot,
                last_actions=last,
            )
            pred = pred.to(dtype=torch.float32)
            loss = _compute_draft_training_loss(
                pred_actions=pred,
                target_actions=tgt,
                gamma=float(args.loss_gamma),
                loss_prefix_mode=str(args.loss_prefix_mode),
                gamma_prefix=float(args.loss_gamma_prefix),
                gamma_tail=float(args.loss_gamma_tail),
                tail_weight=float(args.loss_tail_weight),
                beta=float(args.huber_beta),
                prefix_cap=int(args.loss_prefix_cap),
                device=pred.device,
                generator=g,
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            b = int(pred.shape[0])
            sum_loss += float(loss.detach().item()) * float(b)
            diff = (pred - tgt).to(dtype=torch.float32)
            sum_all += float(torch.sqrt(torch.mean(diff * diff, dim=(1, 2))).sum().item())
            count += b

        if use_ddp:
            t = torch.as_tensor([sum_loss, sum_all, float(count)], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            sum_loss, sum_all, count_f = float(t[0].item()), float(t[1].item()), float(t[2].item())
            count = int(count_f)

        mean_loss = (sum_loss / float(count)) if count > 0 else float("nan")
        mean_rms = (sum_all / float(count)) if count > 0 else float("nan")
        if (not use_ddp) or rank == 0:
            print(f"epoch={epoch} train_weighted_huber={mean_loss:.6f} train_rms_all={mean_rms:.6f} samples={count}")

        do_eval = (
            int(args.eval_interval_epochs) > 0
            and len(val_ds) > 0
            and ((epoch + 1) % int(args.eval_interval_epochs) == 0 or (epoch + 1) == int(args.epochs))
        )
        if use_ddp:
            _dist_barrier(device)
        stop = False
        if do_eval and ((not use_ddp) or rank == 0):
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                persistent_workers=int(args.num_workers) > 0,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )
            head_eval = head.module if isinstance(head, torch.nn.parallel.DistributedDataParallel) else head
            metrics = _eval_head_from_cache(
                head_eval,
                val_loader,
                device=device,
                target_source=str(ds.target_source),
                loss_prefix_mode=str(args.loss_prefix_mode),
                gamma=float(args.loss_gamma),
                gamma_prefix=float(args.loss_gamma_prefix),
                gamma_tail=float(args.loss_gamma_tail),
                tail_weight=float(args.loss_tail_weight),
                beta=float(args.huber_beta),
                exec_steps=int(args.eval_exec_steps),
                max_samples=args.eval_max_samples,
                eval_seed_mode=str(args.eval_seed_mode),
                diagnose_samples=int(args.diagnose_samples),
                diagnose_print_steps=int(args.diagnose_print_steps),
                diagnose_history_probe=bool(args.diagnose_history_probe),
            )
            val_loss = float(metrics["weighted_huber"])
            val_rms_exec = float(metrics["rms_exec"])
            val_rms = float(metrics["rms_all"])
            print(
                f"epoch={epoch} val_weighted_huber={val_loss:.6f} val_rms_exec={val_rms_exec:.6f} exec_steps={int(args.eval_exec_steps)} val_rms_all={val_rms:.6f} "
                f"val_rms_pos={metrics['rms_pos']:.6f} val_rms_rot={metrics['rms_rot']:.6f} val_rms_grip={metrics['rms_grip']:.6f} "
                f"val_samples={int(metrics['samples'])}"
            )

            improved = (val_rms_exec + float(args.early_stop_min_delta)) < best_val_rms_exec or (
                abs(val_rms_exec - best_val_rms_exec) <= float(args.early_stop_min_delta) and val_loss < best_val_huber
            )
            if improved:
                best_val_rms_exec = val_rms_exec
                best_val_huber = val_loss
                bad_evals = 0

                head_to_save = head_eval
                out_path = Path(args.out_ckpt)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ckpt = {
                    "draft_head": head_to_save.state_dict(),
                    "meta": {
                        "chunk_m": int(ds.chunk_m),
                        "out_dim": int(ds.out_dim),
                        "img_dim": int(ds.feature_dim),
                        **_draft_head_meta(head_to_save),
                        "cache_run_dir": str(Path(args.cache_run_dir or "")),
                        "target_source": str(ds.target_source),
                        "teacher_noise_mode": str(ds.teacher_noise_mode),
                        "sample_semantics": str(ds.sample_semantics),
                        "loss_prefix_mode": str(args.loss_prefix_mode),
                        "loss_prefix_cap": int(args.loss_prefix_cap),
                        "loss_gamma_prefix": float(args.loss_gamma_prefix),
                        "loss_gamma_tail": float(args.loss_gamma_tail),
                        "loss_tail_weight": float(args.loss_tail_weight),
                        "best_val_weighted_huber": float(best_val_huber),
                        "best_val_rms_exec": float(best_val_rms_exec),
                        "best_val_rms_all": float(val_rms),
                        "eval_exec_steps": int(args.eval_exec_steps),
                    },
                }
                torch.save(ckpt, out_path)
                print(f"wrote_ckpt={out_path}")
            else:
                bad_evals += 1

            if int(args.early_stop_patience_evals) > 0 and bad_evals >= int(args.early_stop_patience_evals):
                stop = True
                print(
                    f"early_stop epoch={epoch} best_val_rms_exec={best_val_rms_exec:.6f} exec_steps={int(args.eval_exec_steps)} "
                    f"best_val_weighted_huber={best_val_huber:.6f}"
                )

        if use_ddp:
            stop_t = torch.as_tensor([1 if stop else 0], device=device, dtype=torch.int64)
            dist.broadcast(stop_t, src=0)
            stop = bool(int(stop_t.item()))
            _dist_barrier(device)

        if stop:
            break

    if ((not use_ddp) or rank == 0) and (len(val_ds) == 0 or int(args.eval_interval_epochs) <= 0):
        head_to_save = head.module if isinstance(head, torch.nn.parallel.DistributedDataParallel) else head
        out_path = Path(args.out_ckpt)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "draft_head": head_to_save.state_dict(),
            "meta": {
                "chunk_m": int(ds.chunk_m),
                "out_dim": int(ds.out_dim),
                "img_dim": int(ds.feature_dim),
                **_draft_head_meta(head_to_save),
                "cache_run_dir": str(Path(args.cache_run_dir or "")),
                "target_source": str(ds.target_source),
                "teacher_noise_mode": str(ds.teacher_noise_mode),
                "sample_semantics": str(ds.sample_semantics),
                "loss_prefix_mode": str(args.loss_prefix_mode),
                "loss_prefix_cap": int(args.loss_prefix_cap),
                "loss_gamma_prefix": float(args.loss_gamma_prefix),
                "loss_gamma_tail": float(args.loss_gamma_tail),
                "loss_tail_weight": float(args.loss_tail_weight),
            },
        }
        torch.save(ckpt, out_path)
        print(f"wrote_ckpt={out_path}")

    if use_ddp:
        _dist_barrier(device)
        dist.destroy_process_group()


def _eval_from_cache(args: Args) -> None:
    use_ddp, rank, _world_size, _local_rank, device = _setup_ddp()
    torch.manual_seed(int(args.seed) + int(rank))
    np.random.seed(int(args.seed) + int(rank))

    if not use_ddp:
        device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)

    ds = _ShardCacheDataset(Path(args.cache_run_dir or ""))
    _require_supported_cache_semantics(ds)
    _require_cache_compatible_with_head(ds)
    if (not use_ddp) or rank == 0:
        print(
            f"cache_target_source={ds.target_source} teacher_noise_mode={ds.teacher_noise_mode} "
            f"sample_semantics={ds.sample_semantics}"
        )
    split_seed = int(args.seed if args.split_seed is None else args.split_seed)
    train_idx, val_idx, n_train_eps, n_val_eps, n_use = _cache_build_split_indices(
        ds,
        max_samples=args.max_samples,
        val_frac=float(args.val_frac),
        split_seed=split_seed,
    )

    if args.eval_split == "all":
        eval_idx = list(range(int(n_use)))
        split_name = "all"
    elif args.eval_split == "train":
        eval_idx = train_idx
        split_name = "train"
    else:
        eval_idx = val_idx
        split_name = "val"

    eval_ds = torch.utils.data.Subset(ds, eval_idx)

    if ((not use_ddp) or rank == 0):
        print(
            f"cache_samples_total={n_use} eval_split={split_name} eval_samples={len(eval_ds)} "
            f"train_episodes={n_train_eps} val_episodes={n_val_eps} val_frac={float(args.val_frac):.3f} split_seed={split_seed}"
        )

        ckpt_path = _resolve_ckpt_path(args)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        meta, sd = _extract_draft_ckpt_meta_state_dict(ckpt)
        head = _build_draft_head(
            img_dim=int(ds.feature_dim),
            chunk_m=int(ds.chunk_m),
            out_dim=int(ds.out_dim),
            device=device,
            state_dict=sd,
            meta=meta,
        )
        head.load_state_dict(sd, strict=True)

        loader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            persistent_workers=int(args.num_workers) > 0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        metrics = _eval_head_from_cache(
            head,
            loader,
            device=device,
            target_source=str(ds.target_source),
            loss_prefix_mode=str(args.loss_prefix_mode),
            gamma=float(args.loss_gamma),
            gamma_prefix=float(args.loss_gamma_prefix),
            gamma_tail=float(args.loss_gamma_tail),
            tail_weight=float(args.loss_tail_weight),
            beta=float(args.huber_beta),
            exec_steps=int(args.eval_exec_steps),
            max_samples=args.eval_max_samples,
            eval_seed_mode=str(args.eval_seed_mode),
            diagnose_samples=int(args.diagnose_samples),
            diagnose_print_steps=int(args.diagnose_print_steps),
            diagnose_history_probe=bool(args.diagnose_history_probe),
        )
        print(f"ckpt={ckpt_path}")
        print(f"samples={int(metrics['samples'])}")
        print(f"weighted_huber={metrics['weighted_huber']:.6f}")
        print(f"rms_exec={metrics['rms_exec']:.6f} exec_steps={int(args.eval_exec_steps)}")
        print(f"rms_all={metrics['rms_all']:.6f}")
        print(f"rms_pos={metrics['rms_pos']:.6f}")
        print(f"rms_rot={metrics['rms_rot']:.6f}")
        print(f"rms_grip={metrics['rms_grip']:.6f}")

    if use_ddp:
        _dist_barrier(device)
        dist.destroy_process_group()


def main(args: Args) -> None:
    if args.cache_run_dir is None:
        raise ValueError("cache_run_dir is required; image-path training/eval has been removed")
    if args.mode == "eval":
        _eval_from_cache(args)
    else:
        _train_from_cache(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
