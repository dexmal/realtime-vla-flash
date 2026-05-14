import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import jax
import numpy as np
import torch
import torch.distributed as dist
import tqdm
import tyro
from safetensors.torch import save_file as _save_safetensors
from safetensors.torch import load_file as _load_safetensors

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.video_utils import decode_video_frames

from openpi.models import model as _model
from openpi.models_pytorch.spec_pi0_pytorch import SpecArgs
from openpi.models_pytorch.spec_pi0_pytorch import SpecPI0Pytorch
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import image_tools

_SLIDING_CACHE_SAMPLE_SEMANTICS = "sliding_chunk_shift"


def _chw01_to_hwc_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().to(dtype=torch.float32).clamp(0.0, 1.0)
    x_u8 = (x * 255.0).round().to(dtype=torch.uint8)
    return x_u8.permute(1, 2, 0).cpu().numpy()


def _maybe_patch_get_query_indices(ds: LeRobotDataset) -> None:
    if ds.episodes is None or ds.delta_indices is None:
        return

    ep_to_local = {int(ep): i for i, ep in enumerate(ds.episodes)}
    orig_get_query_indices = ds._get_query_indices  # type: ignore[attr-defined]

    def _get_query_indices_patched(idx: int, ep_idx: int):
        local_ep = ep_to_local.get(int(ep_idx), 0)
        return orig_get_query_indices(idx, local_ep)

    ds._get_query_indices = _get_query_indices_patched  # type: ignore[method-assign]


def _is_video_decode_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        exc_type = type(current)
        message = str(current)
        if exc_type.__module__.startswith("av.") and exc_type.__name__ == "InvalidDataError":
            return True
        if "Invalid data found when processing input" in message or "avcodec_send_packet()" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _to_int_scalar(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _to_float_scalar(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def _diagnose_video_decode_failure(ds: LeRobotDataset, idx: int) -> dict[str, Any]:
    raw = ds.hf_dataset[int(idx)]
    ep_idx = _to_int_scalar(raw["episode_index"])
    task_name: str | None = None
    if "task_index" in raw:
        task_idx = _to_int_scalar(raw["task_index"])
        tasks = getattr(ds.meta, "tasks", ())
        if 0 <= task_idx < len(tasks):
            task_name = str(tasks[task_idx])
    timestamp = _to_float_scalar(raw["timestamp"])
    query_timestamps = {str(k): [float(timestamp)] for k in getattr(ds.meta, "video_keys", ())}
    if getattr(ds, "delta_indices", None) is not None:
        query_indices, _padding = ds._get_query_indices(int(idx), int(ep_idx))
        query_timestamps = ds._get_query_timestamps(timestamp, query_indices)

    failed_videos: list[dict[str, str]] = []
    for vid_key in getattr(ds.meta, "video_keys", ()):
        video_path = Path(ds.root) / str(ds.meta.get_video_file_path(ep_idx, vid_key))
        timestamps = list(query_timestamps.get(vid_key, [float(timestamp)]))
        try:
            decode_video_frames(video_path, timestamps, ds.tolerance_s, ds.video_backend)
        except Exception as video_exc:  # pragma: no cover - exercised in tests with monkeypatch
            failed_videos.append(
                {
                    "video_key": str(vid_key),
                    "video_path": str(video_path),
                    "error": f"{type(video_exc).__name__}: {video_exc}",
                }
            )

    return {
        "episode_index": int(ep_idx),
        "task_name": task_name,
        "timestamp": float(timestamp),
        "failed_videos": failed_videos,
    }


def _load_dataset_item_or_skip(ds: LeRobotDataset, idx: int, *, rank: int) -> dict[str, Any] | None:
    try:
        return ds[int(idx)]
    except Exception as exc:
        if not _is_video_decode_error(exc):
            raise

        detail = _diagnose_video_decode_failure(ds, int(idx))
        summary = (
            f"skip_corrupt_video rank={int(rank)} dataset_index={int(idx)} "
            f"episode={int(detail['episode_index'])} timestamp={float(detail['timestamp']):.6f} "
            f"error={type(exc).__name__}: {exc}"
        )
        if detail.get("task_name"):
            summary += f" task={detail['task_name']}"
        lines = [summary]
        failed_videos = list(detail.get("failed_videos", []))
        if failed_videos:
            for failed in failed_videos:
                lines.append(
                    f"failed_video key={failed['video_key']} path={failed['video_path']} error={failed['error']}"
                )
        tqdm.tqdm.write("\n".join(lines))
        return None


def _torch_dtype(name: str) -> torch.dtype:
    name_n = str(name).strip().lower()
    if name_n in {"fp16", "float16", "f16"}:
        return torch.float16
    if name_n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name_n in {"fp32", "float32", "f32"}:
        return torch.float32
    raise ValueError(f"unsupported cache_dtype={name!r} (expected fp16|bf16|fp32)")


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


def _format_action_targets(
    actions: torch.Tensor,
    *,
    chunk_m: int,
    out_dim: int,
) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError(f"expected actions to be (B,H,D), got shape={tuple(actions.shape)}")

    b = int(actions.shape[0])
    m = int(max(1, int(chunk_m)))
    o = int(max(1, int(out_dim)))
    out = torch.zeros((b, m, o), device=actions.device, dtype=torch.float32)

    copy_h = int(min(m, int(actions.shape[1])))
    copy_d = int(min(o, int(actions.shape[2])))
    if copy_h > 0 and copy_d > 0:
        out[:, :copy_h, :copy_d] = actions[:, :copy_h, :copy_d].to(dtype=torch.float32)
    if copy_h < m and copy_h > 0:
        out[:, copy_h:, :] = out[:, copy_h - 1 : copy_h, :]
    return out


@torch.inference_mode()
def _teacher_targets_from_observation(
    spec_model: SpecPI0Pytorch,
    observation: _model.Observation,
    *,
    device: torch.device,
    chunk_m: int,
    out_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images, img_masks, lang_tokens, lang_masks, robot_state = spec_model._preprocess_observation(  # noqa: SLF001
        observation, train=False
    )
    images_t = tuple(images)
    img_masks_t = tuple(img_masks)
    prefix_embs, prefix_pad_masks, prefix_att_masks = spec_model._encoder_stage(  # noqa: SLF001
        images_t, img_masks_t, lang_tokens, lang_masks
    )
    past_key_values = spec_model._vlm_prefill_stage(prefix_embs, prefix_pad_masks, prefix_att_masks)  # noqa: SLF001

    bsize = int(robot_state.shape[0])
    zero_noise = torch.zeros(
        (bsize, int(spec_model.config.action_horizon), int(spec_model.config.action_dim)),
        device=device,
        dtype=torch.float32,
    )
    teacher_actions = spec_model._full_action_stage(  # noqa: SLF001
        robot_state,
        prefix_pad_masks,
        past_key_values,
        zero_noise,
        int(spec_model.spec_args.full_num_steps),
    ).to(dtype=torch.float32)
    return (
        prefix_embs.detach().to(dtype=torch.float32),
        prefix_pad_masks.detach().to(dtype=torch.bool),
        prefix_att_masks.detach().to(dtype=torch.bool),
        robot_state.detach().to(dtype=torch.float32),
        _format_action_targets(teacher_actions, chunk_m=chunk_m, out_dim=out_dim),
    )


@torch.inference_mode()
def _targets_from_observation(
    spec_model: SpecPI0Pytorch,
    observation: _model.Observation,
    *,
    device: torch.device,
    target_source: str,
    chunk_m: int,
    out_dim: int,
    gt_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if str(target_source) == "teacher_zero_noise":
        return _teacher_targets_from_observation(
            spec_model,
            observation,
            device=device,
            chunk_m=chunk_m,
            out_dim=out_dim,
        )
    if str(target_source) != "gt":
        raise ValueError(f"unsupported target_source={target_source!r}")
    if gt_targets is None:
        raise ValueError("gt_targets is required when target_source='gt'")

    images, img_masks, lang_tokens, lang_masks, robot_state = spec_model._preprocess_observation(  # noqa: SLF001
        observation, train=False
    )
    images_t = tuple(images)
    img_masks_t = tuple(img_masks)
    prefix_embs, prefix_pad_masks, prefix_att_masks = spec_model._encoder_stage(  # noqa: SLF001
        images_t, img_masks_t, lang_tokens, lang_masks
    )
    return (
        prefix_embs.detach().to(dtype=torch.float32),
        prefix_pad_masks.detach().to(dtype=torch.bool),
        prefix_att_masks.detach().to(dtype=torch.bool),
        robot_state.detach().to(dtype=torch.float32),
        _format_action_targets(gt_targets.to(dtype=torch.float32), chunk_m=chunk_m, out_dim=out_dim),
    )


@dataclasses.dataclass
class _SlidingWindowState:
    history_len: int
    action_dim: int
    episode_index: int | None = None
    last_actions: torch.Tensor | None = None
    prev_target_chunk: torch.Tensor | None = None

    def reset(self, *, episode_index: int) -> None:
        self.episode_index = int(episode_index)
        self.last_actions = torch.zeros((int(self.history_len), int(self.action_dim)), dtype=torch.float32, device="cpu")
        self.prev_target_chunk = None


@dataclasses.dataclass
class _RankResumeState:
    next_shard_id: int
    processed_indices: set[int]
    complete_shards: list[dict[str, Any]]
    sliding_state: _SlidingWindowState


def _make_shifted_seed_actions(
    prev_target_chunk: torch.Tensor | None,
    *,
    seed_len: int,
    action_dim: int,
) -> torch.Tensor:
    seed = torch.zeros((int(seed_len), int(action_dim)), dtype=torch.float32, device="cpu")
    if prev_target_chunk is None:
        return seed
    if prev_target_chunk.ndim != 2 or int(prev_target_chunk.shape[1]) != int(action_dim):
        raise ValueError(
            f"prev_target_chunk must be (M,D) with D={int(action_dim)}, got {tuple(prev_target_chunk.shape)}"
        )

    shifted = prev_target_chunk[1:, :]
    if int(shifted.shape[0]) <= 0:
        shifted = prev_target_chunk[:1, :]
    take = int(min(int(seed_len), int(shifted.shape[0])))
    if take > 0:
        seed[:take, :] = shifted[:take, :].to(dtype=torch.float32)
    if take < int(seed_len) and take > 0:
        seed[take:, :] = seed[take - 1 : take, :]
    return seed


def _scan_sliding_window_inputs(
    *,
    episode_indices: list[int],
    target_chunks: torch.Tensor,
    state: _SlidingWindowState,
    seed_len: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_chunks.ndim != 3:
        raise ValueError(f"target_chunks must be (B,M,D), got {tuple(target_chunks.shape)}")
    if int(target_chunks.shape[2]) != int(state.action_dim):
        raise ValueError(
            f"target_chunks action dim must match state.action_dim={int(state.action_dim)}, got {int(target_chunks.shape[2])}"
        )

    batch_last: list[torch.Tensor] = []
    batch_seed: list[torch.Tensor] = []
    for batch_idx, ep in enumerate(episode_indices):
        if state.episode_index is None or int(ep) != int(state.episode_index) or state.last_actions is None:
            state.reset(episode_index=int(ep))

        assert state.last_actions is not None
        batch_last.append(state.last_actions.clone())
        batch_seed.append(
            _make_shifted_seed_actions(
                state.prev_target_chunk,
                seed_len=int(seed_len),
                action_dim=int(state.action_dim),
            )
        )

        current_chunk = target_chunks[int(batch_idx)].detach().to(device="cpu", dtype=torch.float32).clone()
        if int(state.history_len) > 0:
            executed = current_chunk[:1, :]
            state.last_actions = torch.cat([state.last_actions[1:, :], executed], dim=0)
        state.prev_target_chunk = current_chunk

    return torch.stack(batch_last, dim=0), torch.stack(batch_seed, dim=0)


def _make_run_dir(cache_dir: str, run_spec: dict[str, Any], config_name: str) -> Path:
    blob = json.dumps(run_spec, sort_keys=True, default=str).encode("utf-8")
    key = hashlib.sha1(blob).hexdigest()[:12]
    return Path(cache_dir) / f"spec_cache_{config_name}_{key}"


def _load_rank_resume_state(run_dir: Path, *, rank: int, history_len: int, action_dim: int) -> _RankResumeState:
    processed_indices: set[int] = set()
    complete_shards: list[dict[str, Any]] = []
    sliding_state = _SlidingWindowState(history_len=int(history_len), action_dim=int(action_dim))
    latest_last_actions: torch.Tensor | None = None
    latest_target_chunk: torch.Tensor | None = None
    latest_episode_index: int | None = None

    shard_id = 0
    while True:
        fname = f"rank{int(rank):03d}_shard{int(shard_id):05d}.safetensors"
        shard_path = run_dir / fname
        if not shard_path.exists():
            break

        try:
            tensors = _load_safetensors(str(shard_path))
            dataset_index = tensors["dataset_index"]
            episode_index = tensors["episode_index"]
            last_actions = tensors["last_actions"]
            targets = tensors["targets"]
        except Exception as exc:
            tqdm.tqdm.write(f"resume_ignore_incomplete_shard rank={int(rank)} shard={fname} error={type(exc).__name__}: {exc}")
            break

        n_samples = int(dataset_index.shape[0])
        complete_shards.append({"path": fname, "num_samples": n_samples, "rank": int(rank)})
        processed_indices.update(int(x) for x in dataset_index.detach().cpu().tolist())
        if n_samples > 0:
            latest_episode_index = int(episode_index[-1].item())
            latest_last_actions = last_actions[-1].detach().to(device="cpu", dtype=torch.float32).clone()
            latest_target_chunk = targets[-1].detach().to(device="cpu", dtype=torch.float32).clone()
        shard_id += 1

    if latest_episode_index is not None:
        sliding_state.episode_index = int(latest_episode_index)
        if latest_last_actions is None or latest_target_chunk is None:
            raise RuntimeError("resume state missing last_actions or target chunk for latest complete sample")
        if int(history_len) > 0:
            executed = latest_target_chunk[:1, : int(action_dim)]
            sliding_state.last_actions = torch.cat([latest_last_actions[1:, : int(action_dim)], executed], dim=0)
        else:
            sliding_state.last_actions = latest_last_actions[:0, : int(action_dim)].clone()
        sliding_state.prev_target_chunk = latest_target_chunk

    return _RankResumeState(
        next_shard_id=int(shard_id),
        processed_indices=processed_indices,
        complete_shards=complete_shards,
        sliding_state=sliding_state,
    )


class _ShardWriter:
    def __init__(
        self,
        *,
        run_dir: Path,
        rank: int,
        shard_size: int,
        cache_dtype: torch.dtype,
        history_len: int,
        chunk_m: int,
        out_dim: int,
        existing_shards: list[dict[str, Any]] | None = None,
        start_shard_id: int = 0,
    ) -> None:
        self._run_dir = run_dir
        self._rank = int(rank)
        self._shard_size = int(shard_size)
        self._dtype = cache_dtype
        self._history_len = int(history_len)
        self._chunk_m = int(chunk_m)
        self._out_dim = int(out_dim)
        self._buf_pos = 0
        self._shard_id = int(start_shard_id)
        self._initialized = False

        self.feature_dim: int | None = None
        self.robot_state_dim: int | None = None
        self.action_dim: int | None = None
        self.shards: list[dict[str, Any]] = list(existing_shards or [])

        self._prefix_embs: torch.Tensor | None = None
        self._prefix_pad_masks: torch.Tensor | None = None
        self._prefix_att_masks: torch.Tensor | None = None
        self._robot: torch.Tensor | None = None
        self._last: torch.Tensor | None = None
        self._seed: torch.Tensor | None = None
        self._tgt: torch.Tensor | None = None
        self._ep: torch.Tensor | None = None
        self._idx: torch.Tensor | None = None

    def _init_buffers(
        self,
        *,
        feature_dim: int,
        robot_state_dim: int,
        action_dim: int,
        prefix_len: int,
        prefix_dim: int,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.robot_state_dim = int(robot_state_dim)
        self.action_dim = int(action_dim)

        n = int(self._shard_size)
        d = int(self.feature_dim)
        r = int(self.robot_state_dim)
        a = int(self.action_dim)
        h = int(self._history_len)
        m = int(self._chunk_m)
        o = int(self._out_dim)
        p = int(prefix_len)
        prefix_d = int(prefix_dim)

        self._prefix_embs = torch.empty((n, p, prefix_d), dtype=self._dtype, device="cpu")
        self._prefix_pad_masks = torch.empty((n, p), dtype=torch.bool, device="cpu")
        self._prefix_att_masks = torch.empty((n, p), dtype=torch.bool, device="cpu")
        self._robot = torch.empty((n, r), dtype=self._dtype, device="cpu")
        self._last = torch.empty((n, h, a), dtype=self._dtype, device="cpu")
        self._seed = torch.empty((n, 2, a), dtype=self._dtype, device="cpu")
        self._tgt = torch.empty((n, m, o), dtype=self._dtype, device="cpu")
        self._ep = torch.empty((n,), dtype=torch.int64, device="cpu")
        self._idx = torch.empty((n,), dtype=torch.int64, device="cpu")
        self._initialized = True

    def _write_shard(self, n: int) -> None:
        if not self._initialized:
            raise RuntimeError("shard writer not initialized")
        if self._prefix_embs is None or self._prefix_pad_masks is None or self._prefix_att_masks is None or self._robot is None:
            raise RuntimeError("buffer not allocated")
        if self._last is None or self._seed is None or self._tgt is None:
            raise RuntimeError("buffer not allocated")
        if self._ep is None or self._idx is None:
            raise RuntimeError("buffer not allocated")

        n_i = int(n)
        fname = f"rank{self._rank:03d}_shard{self._shard_id:05d}.safetensors"
        out_path = self._run_dir / fname
        tensors = {
            "prefix_embs": self._prefix_embs[:n_i].contiguous(),
            "prefix_pad_masks": self._prefix_pad_masks[:n_i].contiguous(),
            "prefix_att_masks": self._prefix_att_masks[:n_i].contiguous(),
            "robot_state": self._robot[:n_i].contiguous(),
            "last_actions": self._last[:n_i].contiguous(),
            "seed_actions": self._seed[:n_i].contiguous(),
            "targets": self._tgt[:n_i].contiguous(),
            "episode_index": self._ep[:n_i].contiguous(),
            "dataset_index": self._idx[:n_i].contiguous(),
        }
        _save_safetensors(tensors, str(out_path))
        self.shards.append({"path": fname, "num_samples": n_i, "rank": self._rank})
        self._shard_id += 1

    def add_batch(self, batch: dict[str, torch.Tensor]) -> None:
        prefix_embs = batch["prefix_embs"]
        prefix_pad_masks = batch["prefix_pad_masks"]
        prefix_att_masks = batch["prefix_att_masks"]
        robot = batch["robot_state"]
        last = batch["last_actions"]
        seed = batch["seed_actions"]
        tgt = batch["targets"]
        ep = batch["episode_index"]
        idx = batch["dataset_index"]

        b = int(prefix_embs.shape[0])
        if b <= 0:
            return

        if not self._initialized:
            self._init_buffers(
                feature_dim=int(prefix_embs.shape[2]),
                robot_state_dim=int(robot.shape[1]),
                action_dim=int(last.shape[2]),
                prefix_len=int(prefix_embs.shape[1]),
                prefix_dim=int(prefix_embs.shape[2]),
            )

        if self._prefix_embs is None or self._prefix_pad_masks is None or self._prefix_att_masks is None or self._robot is None:
            raise RuntimeError("buffer not allocated")
        if self._last is None or self._seed is None or self._tgt is None:
            raise RuntimeError("buffer not allocated")
        if self._ep is None or self._idx is None:
            raise RuntimeError("buffer not allocated")

        src = 0
        while src < b:
            remain = int(self._shard_size - self._buf_pos)
            take = int(min(remain, b - src))
            s0 = int(src)
            s1 = int(src + take)
            d0 = int(self._buf_pos)
            d1 = int(self._buf_pos + take)

            self._prefix_embs[d0:d1].copy_(prefix_embs[s0:s1])
            self._prefix_pad_masks[d0:d1].copy_(prefix_pad_masks[s0:s1])
            self._prefix_att_masks[d0:d1].copy_(prefix_att_masks[s0:s1])
            self._robot[d0:d1].copy_(robot[s0:s1])
            self._last[d0:d1].copy_(last[s0:s1])
            self._seed[d0:d1].copy_(seed[s0:s1])
            self._tgt[d0:d1].copy_(tgt[s0:s1])
            self._ep[d0:d1].copy_(ep[s0:s1])
            self._idx[d0:d1].copy_(idx[s0:s1])

            self._buf_pos = d1
            src = s1

            if self._buf_pos >= int(self._shard_size):
                self._write_shard(int(self._shard_size))
                self._buf_pos = 0

    def finalize(self) -> None:
        if self._initialized and self._buf_pos > 0:
            self._write_shard(self._buf_pos)
            self._buf_pos = 0


@dataclasses.dataclass
class Args:
    # Base policy checkpoint (for input transforms + base weights)
    config: str = "pi0_libero"
    checkpoint_dir: str = "/path/to/pi0_libero_pytorch/"
    device: str | None = None  # ignored under DDP

    # Dataset
    dataset_root: str = "/path/to/libero_spatial_no_noops_1.0.0_lerobot"
    episodes: tuple[int, ...] = ()
    max_samples: int | None = None  # per-process cap when using DDP
    resize_size: int = 224
    video_backend: str = "pyav"

    # Draft targets
    chunk_m: int = 50
    out_dim: int = 7
    max_exec_steps: int = 12
    target_source: Literal["gt", "teacher_zero_noise"] = "teacher_zero_noise"

    # Cache output
    cache_dir: str = "/path/to/spec_cache"
    shard_size: int = 1024
    cache_dtype: str = "fp16"  # fp16|bf16|fp32
    overwrite: bool = False
    resume: bool = False

    # Performance
    batch_size: int = 64
    seed: int = 0


def main(args: Args) -> None:
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")

    use_ddp, rank, world_size, local_rank, device = _setup_ddp()
    torch.manual_seed(int(args.seed) + int(rank))
    np.random.seed(int(args.seed) + int(rank))

    if not use_ddp:
        device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
    else:
        device_str = str(device)

    cache_dtype = _torch_dtype(args.cache_dtype)

    repo_id = os.path.basename(os.path.normpath(args.dataset_root))
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=args.dataset_root)
    all_episodes = list(map(int, meta.episodes))
    selected_episodes = list(map(int, args.episodes)) if args.episodes else all_episodes
    assigned_episodes = selected_episodes[rank::world_size] if use_ddp else selected_episodes

    run_spec = {
        "config": str(args.config),
        "checkpoint_dir": str(args.checkpoint_dir),
        "dataset_root": str(args.dataset_root),
        "episodes": list(map(int, args.episodes)) if args.episodes else "ALL",
        "num_episodes": int(len(selected_episodes)),
        "resize_size": int(args.resize_size),
        "video_backend": str(args.video_backend),
        "chunk_m": int(args.chunk_m),
        "out_dim": int(args.out_dim),
        "max_exec_steps": int(args.max_exec_steps),
        "target_source": str(args.target_source),
        "teacher_noise_mode": "zero" if str(args.target_source) == "teacher_zero_noise" else "none",
        "sample_semantics": _SLIDING_CACHE_SAMPLE_SEMANTICS,
        "cache_dtype": str(args.cache_dtype),
        "shard_size": int(args.shard_size),
    }
    run_dir = _make_run_dir(args.cache_dir, run_spec, config_name=str(args.config))

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.overwrite and not args.resume and any(run_dir.iterdir()):
            raise FileExistsError(f"cache run_dir already exists and is not empty: {run_dir}")
    if use_ddp:
        _dist_barrier(device)

    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint_dir, pytorch_device=device_str)
    base_model = policy._model  # type: ignore[attr-defined]

    spec_model = SpecPI0Pytorch(
        base_model.config,
        spec_args=SpecArgs(
            chunk_m=int(args.chunk_m),
            max_exec_steps=int(min(int(args.max_exec_steps), int(args.chunk_m))),
        ),
    ).to(device)
    spec_model.load_state_dict(base_model.state_dict(), strict=True)
    spec_model.init_spec_modules()
    spec_model.eval()
    history_len = int(getattr(spec_model, "_draft_history_len", 6))
    if rank == 0:
        language_model = spec_model.paligemma_with_expert.paligemma.language_model  # noqa: SLF001
        layer0 = language_model.layers[0]
        lm_config = language_model.config
        torch.save(
            {
                "meta": {
                    "draft_arch": "vlm_block",
                    "draft_input_mode": "prefix_embs",
                    "draft_hidden_size": int(lm_config.hidden_size),
                    "draft_num_heads": int(lm_config.num_attention_heads),
                    "draft_num_kv_heads": int(lm_config.num_key_value_heads),
                    "draft_head_dim": int(lm_config.head_dim),
                },
                "gemma_block": layer0.state_dict(),
            },
            run_dir / "draft_vlm_block_init.pt",
        )

    fps = float(meta.fps)
    horizon = int(train_config.model.action_horizon)
    delta_timestamps = {"action": [t / fps for t in range(horizon)]} if horizon > 0 else None

    ds = LeRobotDataset(
        repo_id=repo_id,
        root=args.dataset_root,
        episodes=assigned_episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )
    _maybe_patch_get_query_indices(ds)

    n_total = len(ds)
    n_use = int(n_total if args.max_samples is None else min(n_total, int(args.max_samples)))
    if n_use <= 0:
        raise ValueError("no samples to cache")

    resume_state: _RankResumeState | None = None
    processed_indices: set[int] = set()
    if args.resume:
        resume_state = _load_rank_resume_state(
            run_dir,
            rank=rank,
            history_len=history_len,
            action_dim=int(args.out_dim),
        )
        processed_indices = set(resume_state.processed_indices)
        if processed_indices:
            tqdm.tqdm.write(
                f"resume_rank={int(rank)} recovered_samples={len(processed_indices)} "
                f"complete_shards={len(resume_state.complete_shards)} next_shard_id={int(resume_state.next_shard_id)}"
            )

    writer = _ShardWriter(
        run_dir=run_dir,
        rank=rank,
        shard_size=int(args.shard_size),
        cache_dtype=cache_dtype,
        history_len=history_len,
        chunk_m=int(args.chunk_m),
        out_dim=int(args.out_dim),
        existing_shards=list(resume_state.complete_shards) if resume_state is not None else None,
        start_shard_id=int(resume_state.next_shard_id) if resume_state is not None else 0,
    )
    sliding_state = (
        resume_state.sliding_state
        if resume_state is not None
        else _SlidingWindowState(history_len=history_len, action_dim=int(args.out_dim))
    )

    batch_inputs_t: list[dict[str, Any]] = []
    batch_gt_targets: list[torch.Tensor] = []
    batch_ep: list[int] = []
    batch_idx: list[int] = []
    skipped_corrupt_samples = 0

    def _flush_batch() -> None:
        if not batch_inputs_t:
            return

        batched_inputs = jax.tree.map(lambda *xs: torch.stack(xs, dim=0), *batch_inputs_t)
        observation = _model.Observation.from_dict(batched_inputs)

        gt_targets = torch.stack(batch_gt_targets, dim=0).to(device=device, dtype=torch.float32) if batch_gt_targets else None
        with torch.inference_mode():
            prefix_embs, prefix_pad_masks, prefix_att_masks, robot_state, targets = _targets_from_observation(
                spec_model,
                observation,
                device=device,
                target_source=str(args.target_source),
                chunk_m=int(args.chunk_m),
                out_dim=int(args.out_dim),
                gt_targets=gt_targets,
            )
        last_actions, seed_actions = _scan_sliding_window_inputs(
            episode_indices=list(map(int, batch_ep)),
            target_chunks=targets,
            state=sliding_state,
        )

        prefix_embs = prefix_embs.detach().to(device="cpu", dtype=cache_dtype)
        prefix_pad_masks = prefix_pad_masks.detach().to(device="cpu")
        prefix_att_masks = prefix_att_masks.detach().to(device="cpu")
        robot = robot_state.detach().to(device="cpu", dtype=cache_dtype)
        last = last_actions.detach().to(device="cpu", dtype=cache_dtype)
        seed = seed_actions.detach().to(device="cpu", dtype=cache_dtype)
        tgt = targets.detach().to(device="cpu", dtype=cache_dtype)[:, : int(args.chunk_m), : int(args.out_dim)]

        ep_t = torch.tensor(batch_ep, dtype=torch.int64, device="cpu")
        idx_t = torch.tensor(batch_idx, dtype=torch.int64, device="cpu")

        writer.add_batch(
            {
                "prefix_embs": prefix_embs,
                "prefix_pad_masks": prefix_pad_masks,
                "prefix_att_masks": prefix_att_masks,
                "robot_state": robot,
                "last_actions": last,
                "seed_actions": seed,
                "targets": tgt,
                "episode_index": ep_t,
                "dataset_index": idx_t,
            }
        )

    sample_iter = tqdm.trange(
        n_use,
        desc=f"cache[{rank}]",
        disable=bool(rank != 0),
        dynamic_ncols=True,
    )
    for i in sample_iter:
        if int(i) in processed_indices:
            continue
        ex = _load_dataset_item_or_skip(ds, i, rank=rank)
        if ex is None:
            skipped_corrupt_samples += 1
            continue
        ep = int(ex["episode_index"].item()) if hasattr(ex["episode_index"], "item") else int(ex["episode_index"])

        state = np.asarray(ex["observation.state"], dtype=np.float32)
        prompt = str(ex.get("task", ""))

        front = _chw01_to_hwc_uint8(ex["observation.images.image"])
        wrist = _chw01_to_hwc_uint8(ex["observation.images.wrist_image"])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(front, args.resize_size, args.resize_size))
        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))

        actions_raw = ex.get("action", None)
        if actions_raw is None:
            continue
        actions_raw_np = np.asarray(actions_raw, dtype=np.float32)

        raw = {
            "observation/state": state,
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "prompt": prompt,
            "actions": actions_raw_np,
        }
        inputs_np = policy._input_transform(raw)  # type: ignore[attr-defined]
        gt_actions = inputs_np.pop("actions", None)
        if gt_actions is None:
            continue
        gt_actions_t = torch.from_numpy(np.asarray(gt_actions, dtype=np.float32))

        inputs_t = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)).to(device), inputs_np)
        batch_inputs_t.append(inputs_t)
        batch_gt_targets.append(gt_actions_t)
        batch_ep.append(ep)
        batch_idx.append(int(i))

        if len(batch_inputs_t) >= int(max(1, int(args.batch_size))):
            _flush_batch()
            batch_inputs_t = []
            batch_gt_targets = []
            batch_ep = []
            batch_idx = []

    _flush_batch()
    writer.finalize()

    local_shards = writer.shards

    def _infer_dims_from_first_shard(shards: list[dict[str, Any]]) -> tuple[int, int, int] | None:
        if not shards:
            return None
        first_path = run_dir / str(shards[0]["path"])
        tensors = _load_safetensors(str(first_path))
        feat_dim = int(tensors["prefix_embs"].shape[2])
        robot_dim = int(tensors["robot_state"].shape[1])
        act_dim = int(tensors["last_actions"].shape[2])
        return feat_dim, robot_dim, act_dim

    inferred = _infer_dims_from_first_shard(local_shards)
    feature_dim = int(writer.feature_dim or (inferred[0] if inferred else 0))
    robot_state_dim = int(writer.robot_state_dim or (inferred[1] if inferred else 0))
    action_dim = int(writer.action_dim or (inferred[2] if inferred else 0))
    if use_ddp:
        gathered: list[list[dict[str, Any]]] | None = [None for _ in range(world_size)] if rank == 0 else None
        skipped_gathered: list[int] | None = [0 for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_shards, gathered, dst=0)
        dist.gather_object(int(skipped_corrupt_samples), skipped_gathered, dst=0)
        _dist_barrier(device)
        if rank == 0:
            flat = [s for sub in (gathered or []) for s in (sub or [])]
            inferred_flat = _infer_dims_from_first_shard(flat)
            feature_dim0 = int(feature_dim or (inferred_flat[0] if inferred_flat else 0))
            robot_state_dim0 = int(robot_state_dim or (inferred_flat[1] if inferred_flat else 0))
            action_dim0 = int(action_dim or (inferred_flat[2] if inferred_flat else 0))
            skipped_total = int(sum(int(x) for x in (skipped_gathered or [])))
            manifest = {
                "run_spec": run_spec,
                "target_source": str(args.target_source),
                "teacher_noise_mode": "zero" if str(args.target_source) == "teacher_zero_noise" else "none",
                "sample_semantics": _SLIDING_CACHE_SAMPLE_SEMANTICS,
                "draft_arch": "vlm_block",
                "draft_input_mode": "prefix_embs",
                "max_exec_steps": int(args.max_exec_steps),
                "draft_history_len": int(history_len),
                "rank_count": int(world_size),
                "cache_dtype": str(args.cache_dtype),
                "feature_dim": feature_dim0,
                "robot_state_dim": robot_state_dim0,
                "action_dim": action_dim0,
                "chunk_m": int(args.chunk_m),
                "out_dim": int(args.out_dim),
                "total_samples": int(sum(int(s["num_samples"]) for s in flat)),
                "skipped_corrupt_samples": skipped_total,
                "shards": flat,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        manifest = {
            "run_spec": run_spec,
            "target_source": str(args.target_source),
            "teacher_noise_mode": "zero" if str(args.target_source) == "teacher_zero_noise" else "none",
            "sample_semantics": _SLIDING_CACHE_SAMPLE_SEMANTICS,
            "draft_arch": "vlm_block",
            "draft_input_mode": "prefix_embs",
            "max_exec_steps": int(args.max_exec_steps),
            "draft_history_len": int(history_len),
            "rank_count": 1,
            "cache_dtype": str(args.cache_dtype),
            "feature_dim": feature_dim,
            "robot_state_dim": robot_state_dim,
            "action_dim": action_dim,
            "chunk_m": int(args.chunk_m),
            "out_dim": int(args.out_dim),
            "total_samples": int(sum(int(s["num_samples"]) for s in local_shards)),
            "skipped_corrupt_samples": int(skipped_corrupt_samples),
            "shards": local_shards,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if use_ddp:
        _dist_barrier(device)
        dist.destroy_process_group()

    if rank == 0:
        print(f"cache_run_dir={run_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
