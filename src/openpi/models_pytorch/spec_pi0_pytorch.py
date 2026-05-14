import dataclasses
import os
import time
from typing import Any
from typing import Literal

import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.draft import DraftChunkHead

try:
    # HF cache class expected by transformers Gemma forward() in recent versions.
    from transformers.cache_utils import DynamicCache as _DynamicCache
except Exception:  # pragma: no cover
    _DynamicCache = None  # type: ignore[assignment]


def _sync_if_cuda(device: torch.device | str) -> None:
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _time_ms(fn, *, device: torch.device | str) -> tuple[Any, float]:
    _sync_if_cuda(device)
    t0 = time.time()
    out = fn()
    _sync_if_cuda(device)
    t1 = time.time()
    return out, (t1 - t0) * 1000.0


def _expand_batch(x: torch.Tensor, k: int) -> torch.Tensor:
    """Expand batch dimension B -> B*K without allocating (uses expand+reshape)."""
    if k == 1:
        return x
    b = x.shape[0]
    return x.unsqueeze(1).expand(b, k, *x.shape[1:]).reshape(b * k, *x.shape[1:])


def _accepted_prefix_len_from_mask(accept_mask: torch.Tensor) -> torch.Tensor:
    """Convert per-step accept decisions into a prefix length."""
    if accept_mask.ndim != 2:
        raise ValueError(f"expected accept_mask to be (B,H), got shape={tuple(accept_mask.shape)}")
    prefix_ok = torch.cumprod(accept_mask.to(dtype=torch.int64), dim=1)
    return prefix_ok.sum(dim=1, dtype=torch.int64)


def _truncate_accepted_prefix_on_gripper_switch(
    *,
    x0_out: torch.Tensor,
    accepted_prefix_len: torch.Tensor,
    gripper_prev: torch.Tensor | None,
    gripper_switch_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x0_out.ndim != 3:
        raise ValueError(f"expected x0_out to be (B,H,D), got shape={tuple(x0_out.shape)}")
    if accepted_prefix_len.ndim != 1 or int(accepted_prefix_len.shape[0]) != int(x0_out.shape[0]):
        raise ValueError(
            f"accepted_prefix_len must have shape (B,)={(int(x0_out.shape[0]),)}, got {tuple(accepted_prefix_len.shape)}"
        )

    b, h, d = x0_out.shape
    accepted_prefix_len = accepted_prefix_len.to(device=x0_out.device, dtype=torch.int64)
    cut_mask = torch.zeros((b,), device=x0_out.device, dtype=torch.bool)
    if d < 7 or gripper_prev is None:
        return accepted_prefix_len, cut_mask
    if gripper_prev.ndim != 1 or int(gripper_prev.shape[0]) != int(b):
        raise ValueError(f"gripper_prev must have shape (B,)={(int(b),)}, got {tuple(gripper_prev.shape)}")

    prev_values = torch.cat(
        [
            gripper_prev.to(device=x0_out.device, dtype=torch.float32)[:, None],
            x0_out[:, :-1, 6].to(dtype=torch.float32),
        ],
        dim=1,
    )
    curr_values = x0_out[:, :, 6].to(dtype=torch.float32)
    threshold = float(gripper_switch_threshold)
    switch_mask = ((prev_values < threshold) & (curr_values >= threshold)) | (
        (prev_values >= threshold) & (curr_values < threshold)
    )
    step_idx = torch.arange(h, device=x0_out.device, dtype=torch.int64)[None, :]
    active_mask = step_idx < accepted_prefix_len[:, None]
    switch_mask = switch_mask & active_mask
    cut_mask = switch_mask.any(dim=1)
    first_switch_idx = switch_mask.to(dtype=torch.int64).argmax(dim=1)
    truncated_prefix_len = torch.where(cut_mask, first_switch_idx, accepted_prefix_len)
    return truncated_prefix_len.to(dtype=torch.int64), cut_mask


def _detect_verify_gripper_switch_any_k(
    *,
    x0_hat: torch.Tensor,
    gripper_prev: torch.Tensor | None,
    gripper_switch_threshold: float,
    eval_h: int,
) -> torch.Tensor:
    if x0_hat.ndim != 4:
        raise ValueError(f"expected x0_hat to be (B,K,H,D), got shape={tuple(x0_hat.shape)}")

    b, k, h, d = x0_hat.shape
    trigger_mask = torch.zeros((b,), device=x0_hat.device, dtype=torch.bool)
    if d < 7 or gripper_prev is None:
        return trigger_mask
    if gripper_prev.ndim != 1 or int(gripper_prev.shape[0]) != int(b):
        raise ValueError(f"gripper_prev must have shape (B,)={(int(b),)}, got {tuple(gripper_prev.shape)}")

    eval_h2 = int(min(h, max(1, int(eval_h))))
    prev_values = torch.cat(
        [
            gripper_prev.to(device=x0_hat.device, dtype=torch.float32)[:, None, None].expand(-1, k, -1),
            x0_hat[:, :, : max(0, eval_h2 - 1), 6].to(dtype=torch.float32),
        ],
        dim=2,
    )
    curr_values = x0_hat[:, :, :eval_h2, 6].to(dtype=torch.float32)
    threshold = float(gripper_switch_threshold)
    switch_mask = ((prev_values < threshold) & (curr_values >= threshold)) | (
        (prev_values >= threshold) & (curr_values < threshold)
    )
    return switch_mask.any(dim=2).any(dim=1)


def _compute_radius_prefix_acceptance(
    *,
    x0_draft: torch.Tensor,
    x0_hat: torch.Tensor,
    tau_radius: float,
    dist_dims: int,
    eval_h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x0_draft.ndim != 3:
        raise ValueError(f"expected x0_draft to be (B,H,D), got shape={tuple(x0_draft.shape)}")
    if x0_hat.ndim != 4:
        raise ValueError(f"expected x0_hat to be (B,K,H,D), got shape={tuple(x0_hat.shape)}")
    b, h, d = int(x0_draft.shape[0]), int(x0_draft.shape[1]), int(x0_draft.shape[2])
    if int(x0_hat.shape[0]) != b or int(x0_hat.shape[2]) != h or int(x0_hat.shape[3]) != d:
        raise ValueError(f"x0_hat must be (B,K,H,D)={(b,'K',h,d)}, got shape={tuple(x0_hat.shape)}")

    eval_h2 = int(min(h, max(1, int(eval_h))))
    eval_d = int(min(d, int(dist_dims)))
    if d >= 7:
        eval_d = int(min(eval_d, 6))
    if eval_d <= 0:
        raise ValueError(f"dist_dims must be >=1, got dist_dims={dist_dims}")

    diff = x0_hat[:, :, :eval_h2, :eval_d] - x0_draft[:, None, :eval_h2, :eval_d]
    norm_d = torch.tensor(float(eval_d), device=x0_draft.device, dtype=torch.float32).sqrt().clamp_min(1.0)
    dist = torch.linalg.vector_norm(diff, ord=2, dim=3).to(dtype=torch.float32) / norm_d

    ok = dist <= float(tau_radius)
    prefix_mask = ok.to(dtype=torch.int64).cumprod(dim=2)
    prefix_len_k = prefix_mask.sum(dim=2)
    accepted_prefix_len = prefix_len_k.min(dim=1).values.to(dtype=torch.int64)
    return accepted_prefix_len, dist


def _stitch_radius_prefix_output(
    *,
    x0_draft: torch.Tensor,
    x0_tail: torch.Tensor,
    accepted_prefix_len: torch.Tensor,
) -> torch.Tensor:
    if x0_draft.ndim != 3:
        raise ValueError(f"expected x0_draft to be (B,H,D), got shape={tuple(x0_draft.shape)}")
    if x0_tail.ndim != 3 or tuple(x0_tail.shape) != tuple(x0_draft.shape):
        raise ValueError(f"x0_tail must match x0_draft shape={tuple(x0_draft.shape)}, got {tuple(x0_tail.shape)}")
    if accepted_prefix_len.ndim != 1 or int(accepted_prefix_len.shape[0]) != int(x0_draft.shape[0]):
        raise ValueError(
            f"accepted_prefix_len must have shape (B,)={(int(x0_draft.shape[0]),)}, got {tuple(accepted_prefix_len.shape)}"
        )

    accepted_prefix_len = accepted_prefix_len.to(device=x0_draft.device, dtype=torch.int64)
    idx = torch.arange(int(x0_draft.shape[1]), device=x0_draft.device, dtype=torch.int64)[None, :]
    accept_mask = (idx < accepted_prefix_len[:, None])[:, :, None]
    return torch.where(accept_mask, x0_draft, x0_tail)


def _should_schedule_full_fallback(
    *,
    full_fallback: bool,
    accepted_prefix_len: torch.Tensor,
    gripper_switch_cut_mask: torch.Tensor | None = None,
) -> bool:
    if not bool(full_fallback):
        return False
    zero_accept = bool((accepted_prefix_len <= 0).any().item())
    switch_cut = bool(gripper_switch_cut_mask is not None and gripper_switch_cut_mask.any().item())
    return zero_accept or switch_cut


def _should_run_full_pipeline_round(
    *,
    cache_ready: bool,
    full_fallback: bool,
    pending_full_fallback: bool,
    force_full_each_round: bool = False,
    periodic_full_every_n_draft_rounds: int = 0,
    draft_rounds_since_full: int = 0,
) -> bool:
    periodic_full = int(periodic_full_every_n_draft_rounds) > 0 and int(draft_rounds_since_full) >= int(
        periodic_full_every_n_draft_rounds
    )
    return bool(force_full_each_round) or bool(periodic_full) or (not bool(cache_ready)) or (
        bool(full_fallback) and bool(pending_full_fallback)
    )


def _full_round_accepted_prefix_len(*, action_horizon: int, max_exec_steps: int) -> int:
    return int(min(int(action_horizon), max(1, int(max_exec_steps))))


def _set_legacy_timing_compat_fields(timing: dict[str, float]) -> None:
    timing["enc_priority_mean"] = 0.0
    timing["score_delta"] = 0.0
    timing["score_total"] = 0.0
    timing["suggest_refresh"] = 0.0
    timing["gripper_override_rate"] = 0.0
    timing["gripper_force_rate"] = 0.0
    timing["gripper_reject_rate"] = 0.0


def _make_speculative_metrics(
    *,
    radius_dist: torch.Tensor,
    accepted_prefix_len_mean: torch.Tensor,
    gripper_switch_cut_rate: torch.Tensor,
    scheduled_full_fallback_gripper: torch.Tensor,
    gripper_verify_stop_rate: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            radius_dist.to(dtype=torch.float32),
            accepted_prefix_len_mean.to(dtype=torch.float32),
            gripper_switch_cut_rate.to(dtype=torch.float32),
            scheduled_full_fallback_gripper.to(dtype=torch.float32),
            gripper_verify_stop_rate.to(dtype=torch.float32),
        ]
    ).to(dtype=torch.float32)


def _populate_full_round_timing(
    timing: dict[str, float],
    *,
    verify_mode: str,
    action_horizon: int,
    max_exec_steps: int,
    full_prefill_ms: float,
    full_action_ms: float,
    gripper_verify_enabled: bool,
) -> float:
    accepted_prefix_len = float(
        _full_round_accepted_prefix_len(action_horizon=action_horizon, max_exec_steps=max_exec_steps)
    )
    timing["vlm_prefill_ms"] = float(full_prefill_ms)
    timing["did_prefill"] = 1.0
    timing["draft_ms"] = 0.0
    timing["action_verify_ms"] = 0.0
    timing["used_full_fallback"] = 1.0
    timing["scheduled_full_fallback"] = 0.0
    timing["is_full_pipeline_round"] = 1.0
    timing["include_in_draft_accept_metrics"] = 0.0
    timing["full_fallback_ms"] = float(full_action_ms)
    timing["total_ms"] = float(timing["encoder_ms"] + float(full_prefill_ms) + float(full_action_ms))
    timing["radius_dist"] = float("nan")
    timing["verify_mode_random"] = 1.0 if verify_mode == "random" else 0.0
    timing["accepted_prefix_len_mean"] = accepted_prefix_len
    _set_legacy_timing_compat_fields(timing)
    timing["gripper_switch_cut_rate"] = 0.0
    timing["scheduled_full_fallback_gripper"] = 0.0
    timing["gripper_verify_stop_rate"] = 0.0
    timing["gripper_verify_enabled"] = 1.0 if gripper_verify_enabled else 0.0
    timing["accepted_prefix_len"] = accepted_prefix_len
    return accepted_prefix_len


@dataclasses.dataclass
class SpecArgs:
    #################################################################################################################
    # Draft/Verify configuration (edit these instead of env vars)
    #################################################################################################################
    chunk_m: int = 50
    max_exec_steps: int = 12

    # Verify timesteps: near-terminal / low-noise (smaller t => closer to x0 in x_t = t*x1 + (1-t)*x0).
    t_list: tuple[float, ...] = (0.10, 0.05)

    # Radius threshold for accepting draft actions (normalized RMS over action dims).
    tau_radius: float = 0.3
    dist_dims: int = 7
    verify_mode: Literal["radius", "random"] = "radius"
    random_accept_prob: float = 0.5
    random_seed: int = 0

    # Draft behavior
    draft_history_len: int = 6
    gripper_switch_threshold: float = 0.0
    enable_gripper_verify: bool = True
    enable_gripper_post_verify: bool = True
    gripper_full_window: int = 1

    # Full pipeline fallback (when accepted_prefix_len==0).
    full_fallback: bool = True
    full_num_steps: int = 10
    force_full_each_round: bool = False
    periodic_full_every_n_draft_rounds: int = 0


def expand_past_key_values(past_key_values, k: int):
    if k == 1:
        return past_key_values

    # Cache path (preferred): preserve Cache type for transformers forward().
    if _DynamicCache is not None and isinstance(past_key_values, _DynamicCache):
        legacy_out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(len(past_key_values)):
            key, value = past_key_values[layer_idx]
            legacy_out.append((_expand_batch(key, k), _expand_batch(value, k)))
        return _DynamicCache.from_legacy_cache(tuple(legacy_out))

    # Legacy list/tuple path.
    legacy_out2: list[tuple[torch.Tensor, torch.Tensor]] = []
    for key, value in past_key_values:
        legacy_out2.append((_expand_batch(key, k), _expand_batch(value, k)))
    return legacy_out2


def clone_past_key_values(past_key_values):
    """Deep-clone past_key_values to avoid CUDAGraph output buffer reuse/overwrite."""
    if past_key_values is None:
        return None

    # Cache path (preferred): preserve Cache type for transformers forward().
    if _DynamicCache is not None and isinstance(past_key_values, _DynamicCache):
        legacy_out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(len(past_key_values)):
            key, value = past_key_values[layer_idx]
            legacy_out.append((key.detach().clone(), value.detach().clone()))
        return _DynamicCache.from_legacy_cache(tuple(legacy_out))

    # Legacy list/tuple path.
    legacy_out2: list[tuple[torch.Tensor, torch.Tensor]] = []
    for key, value in past_key_values:
        legacy_out2.append((key.detach().clone(), value.detach().clone()))
    return legacy_out2


class SpecPI0Pytorch(PI0Pytorch):
    """Spec variant: encoder runs normally, VLM prefill is replaced with random KV generation,
    and action stage is replaced with K-way verify over multiple timesteps.
    """

    def __init__(self, config, spec_args: SpecArgs | None = None):
        super().__init__(config)

        self.spec_args = spec_args or SpecArgs()

        # ---- Spec knobs ----
        # Chunking: keep action_horizon unchanged for now; draft/verify mainly care about first m steps.
        self._chunk_m = int(self.spec_args.chunk_m)
        self._max_exec_steps = int(self.spec_args.max_exec_steps)
        self._draft_history_len = int(max(3, int(self.spec_args.draft_history_len)))

        # Verify timesteps: near-terminal / low-noise (smaller t => closer to x0 in x_t = t*x1 + (1-t)*x0).
        self._verify_t_list: list[float] = list(self.spec_args.t_list)
        self._verify_k = int(len(self._verify_t_list))
        # Cache tks as a non-persistent buffer; move/cast per-call for device/dtype.
        self.register_buffer("_verify_tks", torch.tensor(self._verify_t_list, dtype=torch.float32), persistent=False)

        self._tau_radius = float(self.spec_args.tau_radius)
        self._dist_dims = int(self.spec_args.dist_dims)
        self._verify_mode = str(self.spec_args.verify_mode)
        if self._verify_mode not in {"radius", "random"}:
            raise ValueError(f"unsupported verify_mode={self._verify_mode!r}")
        self._random_accept_prob = float(self.spec_args.random_accept_prob)
        if not (0.0 <= self._random_accept_prob <= 1.0):
            raise ValueError(f"random_accept_prob must be in [0,1], got {self._random_accept_prob}")
        self._random_seed = int(self.spec_args.random_seed)
        self._random_generators: dict[str, torch.Generator] = {}

        # KV cache: only full rounds refresh it. Speculative rounds reuse the latest matching cache entry.
        self._past_key_values_cache = None
        self._past_kv_prefix_len: int | None = None
        self._past_kv_batch_size: int | None = None

        # ---- Cross-round state (lazy init on first call / batch-size change) ----
        self._last_actions: torch.Tensor | None = None  # (B,T,D), T=draft_history_len
        self._action_chunk_cache: torch.Tensor | None = None  # (B,H,D) latest planned actions
        self._action_cache_ptr: int = 0  # how many steps have been executed from the cached chunk
        self._draft_rounds_since_full: int = 0
        self._pending_full_fallback: bool = False
        self._gripper_full_rounds_left: int = 0

        # ---- Optional Spec draft head (initialized explicitly after base weights are loaded) ----
        self._draft_head: DraftChunkHead | None = None
        self._draft_predict_actions = None

        # Rebind stages to Spec implementations (and optionally compile them using the same env toggles).
        self._encoder_stage = self._encoder_stage_impl
        self._vlm_prefill_stage = self._vlm_prefill_stage_impl  # overridden below
        self._action_stage = self._action_stage_impl  # verify stage
        self._full_action_stage = self._full_action_stage_impl

        disable_compile = os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "0").lower() in {"1", "true", "yes"}
        if not disable_compile:
            if os.environ.get("OPENPI_COMPILE_ENCODER", "0").lower() in {"1", "true", "yes"}:
                self._encoder_stage = torch.compile(self._encoder_stage_impl, mode="max-autotune")
            if os.environ.get("OPENPI_COMPILE_VLM_PREFILL", "0").lower() in {"1", "true", "yes"}:
                self._vlm_prefill_stage = torch.compile(self._vlm_prefill_stage_impl, mode="max-autotune")
            if os.environ.get("OPENPI_COMPILE_ACTION", "0").lower() in {"1", "true", "yes"}:
                self._action_stage = torch.compile(self._action_stage_impl, mode="max-autotune")
                self._full_action_stage = torch.compile(self._full_action_stage_impl, mode="max-autotune")

    def reset_runtime_state(self, *, force_prefill: bool = True) -> None:
        """Reset cross-call speculative state.

        Use this at episode boundaries to avoid carrying action/KV history across trajectories.
        """
        del force_prefill
        if hasattr(self, "_score_total"):
            delattr(self, "_score_total")
        self._last_actions = None
        self._action_chunk_cache = None
        self._action_cache_ptr = 0
        self._draft_rounds_since_full = 0
        self._pending_full_fallback = False
        self._gripper_full_rounds_left = 0
        # Clear KV cache; the next call will run a full round because no valid cache is available.
        self._past_key_values_cache = None
        self._past_kv_prefix_len = None
        self._past_kv_batch_size = None

    def _accept_full_round_actions(
        self,
        actions: torch.Tensor,
        *,
        past_key_values,
        prefix_len: int,
        batch_size: int,
    ) -> None:
        self._draft_rounds_since_full = 0
        if int(self._gripper_full_rounds_left) > 0:
            self._gripper_full_rounds_left = max(0, int(self._gripper_full_rounds_left) - 1)
            self._pending_full_fallback = bool(self._gripper_full_rounds_left > 0)
        else:
            self._pending_full_fallback = False
        self._past_key_values_cache = past_key_values
        self._past_kv_prefix_len = int(prefix_len)
        self._past_kv_batch_size = int(batch_size)
        self._action_chunk_cache = actions.detach().clone()
        self._action_cache_ptr = 0

    def _schedule_gripper_full_fallback(self) -> None:
        self._pending_full_fallback = True
        self._gripper_full_rounds_left = max(1, int(self.spec_args.gripper_full_window))

    def _bind_draft_runtime(self) -> None:
        if self._draft_head is None:
            self._draft_predict_actions = None
            return

        draft_predict_actions = self._draft_head.forward
        disable_compile = os.environ.get("OPENPI_DISABLE_TORCH_COMPILE", "0").lower() in {"1", "true", "yes"}
        if not disable_compile and os.environ.get("OPENPI_COMPILE_DRAFT", "0").lower() in {"1", "true", "yes"}:
            draft_predict_actions = torch.compile(draft_predict_actions, mode="max-autotune")
        self._draft_predict_actions = draft_predict_actions

    def _make_draft_head(
        self,
        *,
        img_dim: int,
        device: torch.device,
        state_dict: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> DraftChunkHead:
        meta = dict(meta or {})
        state_dict = state_dict or {}
        layer0 = None
        lm_config = None
        paligemma_with_expert = getattr(self, "paligemma_with_expert", None)
        if paligemma_with_expert is not None:
            paligemma = getattr(paligemma_with_expert, "paligemma", None)
            language_model = getattr(paligemma, "language_model", None)
            if language_model is not None:
                lm_config = getattr(language_model, "config", None)
                layers = getattr(language_model, "layers", None)
                if layers is not None and len(layers) > 0:
                    layer0 = layers[0]

        runtime_dtype = torch.float32
        parameters = getattr(self, "parameters", None)
        if callable(parameters):
            try:
                runtime_dtype = next(parameters()).dtype
            except StopIteration:
                pass
        hidden_dim = int(
            state_dict.get(
                "_gemma_block.mlp.gate_proj.weight",
                torch.empty((int(getattr(lm_config, "intermediate_size", 256)), 1)),
            ).shape[0]
        )
        num_heads = int(
            meta.get("draft_num_heads", getattr(lm_config, "num_attention_heads", DraftChunkHead._resolve_num_heads(int(img_dim))))
        )
        num_kv_heads = int(meta.get("draft_num_kv_heads", getattr(lm_config, "num_key_value_heads", 1)))
        head_dim = int(meta.get("draft_head_dim", getattr(lm_config, "head_dim", max(1, int(img_dim) // max(1, num_heads)))))
        head = DraftChunkHead(
            img_dim=img_dim,
            chunk_m=int(self._chunk_m),
            hidden_dim=int(hidden_dim),
            out_dim=7,
            num_heads=int(num_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            dtype=runtime_dtype,
            attn_implementation="sdpa",
        ).to(device=device, dtype=runtime_dtype)
        if not state_dict and layer0 is not None:
            head.init_from_vlm_layer(layer0)
        return head

    def init_spec_modules(self) -> None:
        """Initialize optional Spec modules after base weights have been loaded.

        This is intentionally separate from __init__ so callers can load PI0 weights with strict=True
        before adding new parameters.
        """
        dev = next(self.parameters()).device
        if self._draft_head is None:
            vision_config = getattr(self.paligemma_with_expert.paligemma.config, "vision_config", None)
            img_dim = int(getattr(vision_config, "projection_dim", 2048))
            self._draft_head = self._make_draft_head(img_dim=img_dim, device=dev)
        self._bind_draft_runtime()

    def load_draft_head(self, path: str) -> dict[str, Any]:
        ckpt = torch.load(path, map_location="cpu")
        if self._draft_head is None:
            raise ValueError("Draft head not initialized. Call `init_spec_modules()` first.")

        meta: dict[str, Any] = {}
        sd: Any = None
        if isinstance(ckpt, dict) and "draft_head" in ckpt:
            sd = ckpt.get("draft_head", None)
            meta = dict(ckpt.get("meta", {}) or {})
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            raise ValueError("draft checkpoint must be a state_dict or a dict with key `draft_head`")
        if not isinstance(sd, dict):
            raise ValueError("draft_head state_dict missing or invalid")

        if self._draft_head is None:
            raise ValueError("Draft head not initialized. Call `init_spec_modules()` first.")
        vision_config = getattr(self.paligemma_with_expert.paligemma.config, "vision_config", None)
        img_dim = int(getattr(vision_config, "projection_dim", 2048))
        dev = next(self.parameters()).device
        self._draft_head = self._make_draft_head(
            img_dim=img_dim,
            device=dev,
            state_dict=sd,
            meta=meta,
        )

        self._draft_head.load_state_dict(sd, strict=True)
        self._bind_draft_runtime()
        return {"meta": meta}

    def _compute_draft(
        self,
        *,
        noise: torch.Tensor,
        prefix_embs: torch.Tensor | None = None,
        prefix_pad_masks: torch.Tensor | None = None,
        prefix_att_masks: torch.Tensor | None = None,
        robot_state: torch.Tensor,
        last_actions: torch.Tensor,
    ) -> torch.Tensor:
        if self._draft_head is None:
            raise ValueError("A learned draft head must be loaded before runtime speculative inference.")

        b, h, d = int(noise.shape[0]), int(noise.shape[1]), int(noise.shape[2])
        if d < 7:
            raise ValueError(f"draft head requires action_dim>=7, got {d}")
        if prefix_embs is None or prefix_pad_masks is None or prefix_att_masks is None:
            raise ValueError("draft head requires prefix_embs, prefix_pad_masks, and prefix_att_masks inputs")
        draft_predict_actions = getattr(self, "_draft_predict_actions", None)
        if draft_predict_actions is None:
            draft_predict_actions = self._draft_head.forward
        chunk = draft_predict_actions(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            robot_state=robot_state,
            last_actions=last_actions,
        ).to(dtype=torch.float32)

        x0 = torch.zeros((b, h, d), device=noise.device, dtype=torch.float32)
        m = int(min(int(self._chunk_m), h))
        x0[:, :m, :7] = chunk[:, :m, :]
        if h > m and m > 0:
            x0[:, m:, :7] = x0[:, m - 1 : m, :7].expand(b, h - m, 7)

        return x0

    # ---- Spec stage overrides ----

    def _encoder_stage_impl(self, images, img_masks, lang_tokens, lang_masks):
        # Spec encoder: keep original prefix embeddings for VLM and draft.
        # Keep the compiled return flat: nested tuples can trip CUDAWarmupNode weakref bookkeeping in
        # torch._inductor.cudagraph_trees during warmup.
        embs = []
        pad_masks = []
        att_masks: list[int] = []

        for view_idx, (img, img_mask) in enumerate(zip(images, img_masks, strict=True)):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * int(num_img_embs)

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * (float(lang_emb_dim) ** 0.5)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * int(lang_emb.shape[1])

        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks_1d = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
        bsize = int(prefix_pad_masks.shape[0])
        prefix_att_masks = prefix_att_masks_1d[None, :].expand(bsize, int(prefix_att_masks_1d.shape[0]))
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def _vlm_prefill_stage_impl(self, prefix_embs, prefix_pad_masks, prefix_att_masks):
        # Real VLM prefill only. Caching / refresh policy is handled at the caller level.
        return super()._vlm_prefill_stage_impl(prefix_embs, prefix_pad_masks, prefix_att_masks)

    def _get_cached_past_key_values(self, prefix_pad_masks):
        b = int(prefix_pad_masks.shape[0])
        prefix_len = int(prefix_pad_masks.shape[1])

        cache_valid = self._past_key_values_cache is not None
        cache_valid = cache_valid and (self._past_kv_prefix_len == prefix_len)
        cache_valid = cache_valid and (self._past_kv_batch_size == b)
        if not cache_valid:
            return None
        return self._past_key_values_cache

    def _action_stage_impl(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        last_gripper: torch.Tensor | None,
    ):
        if self._verify_mode == "random":
            return self._action_stage_random_impl(noise, x0_draft, last_gripper)
        # Spec verify stage (BK-parallel), excluding draft/proposal time.
        k = self._verify_k
        b, h, d = noise.shape

        # 1) build K timesteps (near-terminal list)
        tks = self._verify_tks.to(device=noise.device, dtype=noise.dtype)
        if int(tks.numel()) != int(k):
            tks = tks[:k]

        # 2) build x_tk (B,K,H,D) -> (B*K,H,D)
        t = tks[None, :, None, None]  # (1,K,1,1)
        noise_bk = noise[:, None, :, :]  # (B,1,H,D)
        x0_bk = x0_draft[:, None, :, :]
        x_t_bk = (t * noise_bk + (1.0 - t) * x0_bk).reshape(b * k, h, d)
        timestep_bk = tks[None, :].expand(b, k).reshape(b * k)

        # 3) expand conditioning tensors to BK
        state_bk = _expand_batch(state, k)
        prefix_pad_masks_bk = _expand_batch(prefix_pad_masks, k)
        past_key_values_bk = expand_past_key_values(past_key_values, k)

        # 4) run denoise_step in one BK batch
        v_t_bk = self.denoise_step(
            state_bk,
            prefix_pad_masks_bk,
            past_key_values_bk,
            x_t_bk,
            timestep_bk,
        )  # (B*K,H,D)

        # 5) x0_hat_k = x_t - t_k * v(t_k)
        t_bk = timestep_bk[:, None, None]  # (B*K,1,1)
        x0_hat_flat = x_t_bk - t_bk * v_t_bk
        x0_hat = x0_hat_flat.reshape(b, k, h, d)
        # NOTE: no clamping here to stay aligned with training-time behavior.

        # Only verify the portion of the chunk that will actually be executed before the next replan.
        eval_h = int(min(h, max(1, int(self._max_exec_steps))))
        accepted_prefix_len, dist = _compute_radius_prefix_acceptance(
            x0_draft=x0_draft,
            x0_hat=x0_hat,
            tau_radius=float(self._tau_radius),
            dist_dims=int(self._dist_dims),
            eval_h=int(eval_h),
        )
        x0_tail = x0_hat.mean(dim=1)
        x0_out = _stitch_radius_prefix_output(
            x0_draft=x0_draft,
            x0_tail=x0_tail,
            accepted_prefix_len=accepted_prefix_len,
        )
        gripper_force_mask = torch.zeros((b, eval_h), device=noise.device, dtype=torch.bool)
        gripper_reject_mask = torch.zeros((b, eval_h), device=noise.device, dtype=torch.bool)
        gripper_switch_cut_mask = torch.zeros((b,), device=noise.device, dtype=torch.bool)
        gripper_verify_stop_mask = torch.zeros((b,), device=noise.device, dtype=torch.bool)
        if bool(self.spec_args.enable_gripper_verify):
            gripper_verify_stop_mask = _detect_verify_gripper_switch_any_k(
                x0_hat=x0_hat,
                gripper_prev=last_gripper,
                gripper_switch_threshold=float(self.spec_args.gripper_switch_threshold),
                eval_h=int(eval_h),
            )
            accepted_prefix_len = torch.where(
                gripper_verify_stop_mask,
                torch.zeros_like(accepted_prefix_len),
                accepted_prefix_len,
            )
            x0_out = torch.where(gripper_verify_stop_mask[:, None, None], x0_tail, x0_out)
        if bool(self.spec_args.enable_gripper_post_verify):
            accepted_after_cut, gripper_switch_cut_mask = _truncate_accepted_prefix_on_gripper_switch(
                x0_out=x0_out,
                accepted_prefix_len=accepted_prefix_len,
                gripper_prev=last_gripper,
                gripper_switch_threshold=float(self.spec_args.gripper_switch_threshold),
            )
            accepted_prefix_len = torch.where(gripper_verify_stop_mask, accepted_prefix_len, accepted_after_cut)
            gripper_switch_cut_mask = gripper_switch_cut_mask & (~gripper_verify_stop_mask)

        # Keep the compiled metrics tensor restricted to live speculative signals.
        metrics = _make_speculative_metrics(
            radius_dist=dist.mean(),
            accepted_prefix_len_mean=accepted_prefix_len.to(dtype=torch.float32).mean(),
            gripper_switch_cut_rate=gripper_switch_cut_mask.to(dtype=torch.float32).mean(),
            scheduled_full_fallback_gripper=(gripper_switch_cut_mask | gripper_verify_stop_mask)
            .any()
            .to(dtype=torch.float32),
            gripper_verify_stop_rate=gripper_verify_stop_mask.to(dtype=torch.float32).mean(),
        )

        return x0_out, metrics, accepted_prefix_len

    def _get_random_generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self._random_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._random_seed)
            self._random_generators[key] = generator
        return generator

    def _action_stage_random_impl(
        self,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        last_gripper: torch.Tensor | None,
    ):
        b, h, _d = noise.shape
        del last_gripper
        eval_h = int(min(h, max(1, int(self._max_exec_steps))))
        generator = self._get_random_generator(noise.device)
        accept_mask = torch.rand((b, eval_h), device=noise.device, generator=generator) < self._random_accept_prob
        accepted_prefix_len = _accepted_prefix_len_from_mask(accept_mask)

        metrics = _make_speculative_metrics(
            radius_dist=torch.zeros((), device=noise.device, dtype=torch.float32),
            accepted_prefix_len_mean=accepted_prefix_len.to(dtype=torch.float32).mean(),
            gripper_switch_cut_rate=torch.zeros((), device=noise.device, dtype=torch.float32),
            scheduled_full_fallback_gripper=torch.zeros((), device=noise.device, dtype=torch.float32),
            gripper_verify_stop_rate=torch.zeros((), device=noise.device, dtype=torch.float32),
        )

        return x0_draft, metrics, accepted_prefix_len

    def _full_action_stage_impl(self, state, prefix_pad_masks, past_key_values, noise, num_steps: int):
        return PI0Pytorch._action_stage_impl(self, state, prefix_pad_masks, past_key_values, noise, num_steps)

    def _sample_actions_impl(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        executed_steps: int | None = None,
        *,
        collect_timing: bool,
    ):
        del num_steps
        if collect_timing and torch.cuda.is_available() and hasattr(torch, "compiler"):
            mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
            if callable(mark):
                mark()

        timing: dict[str, float] | None = {} if collect_timing else None
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Lazy init / resize cross-round state.
        if (
            (self._last_actions is None)
            or (self._last_actions.shape[0] != bsize)
            or (self._last_actions.shape[2] != int(noise.shape[2]))
            or (self._last_actions.device != noise.device)
        ):
            self._last_actions = torch.zeros(
                (bsize, int(self._draft_history_len), int(noise.shape[2])),
                device=noise.device,
                dtype=torch.float32,
            )
        # Advance action cache pointer by executed steps (controller execution between calls).
        if (
            (self._action_chunk_cache is not None)
            and (self._action_chunk_cache.shape[0] == bsize)
            and (self._action_chunk_cache.shape[2] == int(noise.shape[2]))
            and (self._action_chunk_cache.device == noise.device)
        ):
            h = int(self._action_chunk_cache.shape[1])
            step_adv = int(self._max_exec_steps) if executed_steps is None else int(executed_steps)
            step_adv = max(0, min(step_adv, h))
            self._action_cache_ptr = min(int(self._action_cache_ptr) + step_adv, h)
            if self._action_cache_ptr > 0:
                executed_anchor = self._action_chunk_cache[:, self._action_cache_ptr - 1, :].to(dtype=torch.float32)
                self._last_actions = torch.cat(
                    [self._last_actions[:, 1:, :], executed_anchor[:, None, :]], dim=1
                ).detach()

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        images_t = tuple(images)
        img_masks_t = tuple(img_masks)

        if timing is None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._encoder_stage(images_t, img_masks_t, lang_tokens, lang_masks)
        else:
            (prefix_embs, prefix_pad_masks, prefix_att_masks), ms = _time_ms(
                lambda: self._encoder_stage(images_t, img_masks_t, lang_tokens, lang_masks),
                device=device,
            )
            timing["encoder_ms"] = ms
        if timing is not None:
            timing["enc_diff_front"] = 0.0
            timing["enc_diff_wrist"] = 0.0
            timing["enc_diff_max"] = 0.0
            timing["force_prefill_now"] = 0.0

        past_key_values = self._get_cached_past_key_values(prefix_pad_masks)
        run_full_pipeline_round = _should_run_full_pipeline_round(
            cache_ready=past_key_values is not None,
            full_fallback=bool(self.spec_args.full_fallback),
            pending_full_fallback=bool(self._pending_full_fallback),
            force_full_each_round=bool(self.spec_args.force_full_each_round),
            periodic_full_every_n_draft_rounds=int(self.spec_args.periodic_full_every_n_draft_rounds),
            draft_rounds_since_full=int(self._draft_rounds_since_full),
        )
        if run_full_pipeline_round:
            full_steps = int(max(1, int(self.spec_args.full_num_steps)))
            prefix_len = int(prefix_pad_masks.shape[1])
            if timing is None:
                past_kv = clone_past_key_values(self._vlm_prefill_stage(prefix_embs, prefix_pad_masks, prefix_att_masks))
                actions = self._full_action_stage(state, prefix_pad_masks, past_kv, noise, full_steps)
            else:
                past_kv, full_prefill_ms = _time_ms(
                    lambda: clone_past_key_values(self._vlm_prefill_stage(prefix_embs, prefix_pad_masks, prefix_att_masks)),
                    device=device,
                )
                actions, full_action_ms = _time_ms(
                    lambda: self._full_action_stage(state, prefix_pad_masks, past_kv, noise, full_steps),
                    device=device,
                )
            self._accept_full_round_actions(
                actions,
                past_key_values=past_kv,
                prefix_len=prefix_len,
                batch_size=bsize,
            )
            if timing is None:
                return actions
            _populate_full_round_timing(
                timing,
                verify_mode=self._verify_mode,
                action_horizon=noise.shape[1],
                max_exec_steps=self._max_exec_steps,
                full_prefill_ms=full_prefill_ms,
                full_action_ms=full_action_ms,
                gripper_verify_enabled=bool(self.spec_args.enable_gripper_verify),
            )
            return actions, timing

        if timing is not None:
            timing["vlm_prefill_ms"] = 0.0
            timing["did_prefill"] = 0.0

        if timing is None:
            x0 = self._compute_draft(
                noise=noise,
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                robot_state=state,
                last_actions=self._last_actions,
            )
        else:
            x0, ms = _time_ms(
                lambda: self._compute_draft(
                    noise=noise,
                    prefix_embs=prefix_embs,
                    prefix_pad_masks=prefix_pad_masks,
                    prefix_att_masks=prefix_att_masks,
                    robot_state=state,
                    last_actions=self._last_actions,
                ),
                device=device,
            )
            timing["draft_ms"] = ms
        last_gripper = self._last_actions[:, -1, 6].to(dtype=torch.float32) if int(x0.shape[2]) >= 7 else None

        if timing is None:
            actions, metrics, accepted_prefix_len = self._action_stage(state, prefix_pad_masks, past_key_values, noise, x0, last_gripper)
        else:
            (actions, metrics, accepted_prefix_len), ms = _time_ms(
                lambda: self._action_stage(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    noise,
                    x0,
                    last_gripper,
                ),
                device=device,
            )
            timing["action_verify_ms"] = ms

        should_schedule_full_fallback = _should_schedule_full_fallback(
            full_fallback=bool(self.spec_args.full_fallback),
            accepted_prefix_len=accepted_prefix_len,
            gripper_switch_cut_mask=metrics[3:4].to(dtype=torch.bool),
        )
        if should_schedule_full_fallback:
            if bool(metrics[3].item()):
                self._schedule_gripper_full_fallback()
            else:
                self._pending_full_fallback = True
                self._gripper_full_rounds_left = 0
            if timing is None:
                return actions
        else:
            self._action_chunk_cache = actions.detach().clone()
            self._action_cache_ptr = 0

        self._draft_rounds_since_full = int(self._draft_rounds_since_full) + 1

        if timing is None:
            return actions

        timing["used_full_fallback"] = 0.0
        timing["scheduled_full_fallback"] = 1.0 if should_schedule_full_fallback else 0.0
        timing["is_full_pipeline_round"] = 0.0
        timing["include_in_draft_accept_metrics"] = 1.0
        timing["full_fallback_ms"] = 0.0
        timing["total_ms"] = float(
            timing["encoder_ms"]
            + timing["vlm_prefill_ms"]
            + timing["draft_ms"]
            + timing["action_verify_ms"]
        )
        timing["radius_dist"] = float(metrics[0].item())
        timing["verify_mode_random"] = 1.0 if self._verify_mode == "random" else 0.0
        timing["accepted_prefix_len_mean"] = float(metrics[1].item())
        _set_legacy_timing_compat_fields(timing)
        timing["gripper_switch_cut_rate"] = float(metrics[2].item())
        timing["scheduled_full_fallback_gripper"] = float(metrics[3].item())
        timing["gripper_verify_stop_rate"] = float(metrics[4].item())
        timing["gripper_verify_enabled"] = 1.0 if self.spec_args.enable_gripper_verify else 0.0
        timing["accepted_prefix_len"] = float(accepted_prefix_len.to(dtype=torch.float32).mean().item())
        return actions, timing

    @torch.no_grad()
    def sample_actions_with_timing(self, device, observation, noise=None, num_steps=10, executed_steps: int | None = None):
        """Staged timing: encoder / vlm_prefill / draft / verify (with accepted-prefix)."""
        return self._sample_actions_impl(
            device,
            observation,
            noise=noise,
            num_steps=num_steps,
            executed_steps=executed_steps,
            collect_timing=True,
        )

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, executed_steps: int | None = None):
        # Sampling without timing syncs (for Policy.infer()).
        return self._sample_actions_impl(
            device,
            observation,
            noise=noise,
            num_steps=num_steps,
            executed_steps=executed_steps,
            collect_timing=False,
        )
