import dataclasses
import importlib.util
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

from openpi.models_pytorch.spec_pi0_pytorch import _compute_radius_prefix_acceptance
from openpi.models_pytorch.spec_pi0_pytorch import _detect_verify_gripper_switch_any_k
from openpi.models_pytorch.spec_pi0_pytorch import _make_speculative_metrics
from openpi.models_pytorch.spec_pi0_pytorch import _stitch_radius_prefix_output
from openpi.models_pytorch.spec_pi0_pytorch import _truncate_accepted_prefix_on_gripper_switch


def _spec_triton_env(name: str, default: str) -> str:
    spec_name = f"SPEC_TRITON_{name}"
    legacy_env_name = f"STAR_TRITON_{name}"
    return os.environ.get(spec_name, os.environ.get(legacy_env_name, default))


def _load_pi0_infer_module():
    module_path = Path(__file__).with_name("pi0_infer.py")
    spec = importlib.util.spec_from_file_location("spec_pi0_infer_for_spec", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PI0_INFER = _load_pi0_infer_module()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_ms(fn, *, device: torch.device):
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    out = fn()
    _sync_if_cuda(device)
    return out, (time.perf_counter() - t0) * 1000.0


def _expand_batch(x: torch.Tensor, k: int) -> torch.Tensor:
    if int(k) == 1:
        return x
    b = int(x.shape[0])
    return x.unsqueeze(1).expand(b, int(k), *x.shape[1:]).reshape(b * int(k), *x.shape[1:])


def _matmul_grid(seq_len: int, hidden: int, *, block_n: int, block_m: int) -> tuple[int]:
    return (triton.cdiv(int(seq_len), int(block_n)) * triton.cdiv(int(hidden), int(block_m)),)


def _default_block_size(dim: int) -> int:
    return 64 if int(dim) >= 64 else 32


def _head_block_size(dim: int) -> int:
    block = 1
    while block * 2 <= int(dim) and block * 2 <= 32:
        block *= 2
    return max(1, block)


def _next_power_of_two(dim: int) -> int:
    value = 1
    while value < int(dim):
        value *= 2
    return value


def _kernel_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _canonicalize_linear_weight(weight: torch.Tensor, *, in_features: int, out_features: int) -> torch.Tensor:
    expected_io = (int(in_features), int(out_features))
    expected_oi = (int(out_features), int(in_features))
    if tuple(weight.shape) == expected_io:
        return weight.contiguous()
    if tuple(weight.shape) == expected_oi:
        return weight.transpose(0, 1).contiguous()
    raise ValueError(
        f"linear weight shape={tuple(weight.shape)} is incompatible with in_features={in_features} out_features={out_features}"
    )


def _to_kernel_tensor(x: torch.Tensor, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is None:
        dtype = _kernel_dtype(device)
    return x.to(device=device, dtype=dtype).contiguous()


def _matmul_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float32) @ weight.to(device=x.device, dtype=torch.float32)


def _matmul_bias_fp32(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return _matmul_fp32(x, weight) + bias.to(device=x.device, dtype=torch.float32)


def _rms_norm_fp32(x: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    x = x.to(dtype=torch.float32)
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(eps))


def _apply_gemma_rope_fp32(x: torch.Tensor, rope_weights: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    rows, cols = map(int, x.shape)
    if int(head_dim) <= 0 or cols % int(head_dim) != 0:
        raise ValueError(f"invalid RoPE shape cols={cols} head_dim={head_dim}")
    half_dim = int(head_dim) // 2
    if int(half_dim) * 2 != int(head_dim):
        raise ValueError(f"Gemma RoPE requires even head_dim, got {head_dim}")
    heads = cols // int(head_dim)
    x_view = x.to(dtype=torch.float32).view(rows, heads, int(head_dim))
    rope = rope_weights.to(device=x.device, dtype=torch.float32).view(rows, 1, int(head_dim))
    cos = rope[:, :, :half_dim]
    sin = rope[:, :, half_dim:]
    x_first = x_view[:, :, :half_dim]
    x_second = x_view[:, :, half_dim:]
    out_first = x_first * cos - x_second * sin
    out_second = x_second * cos + x_first * sin
    return torch.cat((out_first, out_second), dim=-1).reshape(rows, cols).contiguous()


def _matmul_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    seq_len, features = map(int, x.shape)
    hidden = int(weight.shape[1])
    x = _to_kernel_tensor(x, device=x.device)
    weight = _to_kernel_tensor(weight, device=x.device)
    if out is None:
        out = torch.empty((seq_len, hidden), device=x.device, dtype=_kernel_dtype(x.device))
    block_n = _default_block_size(seq_len)
    block_m = _default_block_size(hidden)
    block_k = _default_block_size(features)
    _PI0_INFER.matmul_small[_matmul_grid(seq_len, hidden, block_n=block_n, block_m=block_m)](
        x,
        weight,
        out,
        seq_len=seq_len,
        features=features,
        hidden=hidden,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
    )
    return out


def _matmul_bias_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    seq_len, features = map(int, x.shape)
    hidden = int(weight.shape[1])
    x = _to_kernel_tensor(x, device=x.device)
    weight = _to_kernel_tensor(weight, device=x.device)
    bias = _to_kernel_tensor(bias, device=x.device)
    if out is None:
        out = torch.empty((seq_len, hidden), device=x.device, dtype=_kernel_dtype(x.device))
    block_n = _default_block_size(seq_len)
    block_m = _default_block_size(hidden)
    block_k = _default_block_size(features)
    _PI0_INFER.matmul_small_bias[_matmul_grid(seq_len, hidden, block_n=block_n, block_m=block_m)](
        x,
        weight,
        out,
        bias,
        seq_len=seq_len,
        features=features,
        hidden=hidden,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
    )
    return out


def _matmul_residual_kernel(x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    seq_len, features = map(int, x.shape)
    hidden = int(weight.shape[1])
    x = _to_kernel_tensor(x, device=x.device)
    weight = _to_kernel_tensor(weight, device=x.device)
    out = _to_kernel_tensor(residual, device=x.device).clone()
    block_n = _default_block_size(seq_len)
    block_m = _default_block_size(hidden)
    block_k = _default_block_size(features)
    _PI0_INFER.matmul_small_res[_matmul_grid(seq_len, hidden, block_n=block_n, block_m=block_m)](
        x,
        weight,
        out,
        out,
        seq_len=seq_len,
        features=features,
        hidden=hidden,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
    )
    return out


def _rms_matmul_gate_kernel(x: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor) -> torch.Tensor:
    seq_len, features = map(int, x.shape)
    hidden = int(weight1.shape[1])
    x = _to_kernel_tensor(x, device=x.device)
    weight1 = _to_kernel_tensor(weight1, device=x.device)
    weight2 = _to_kernel_tensor(weight2, device=x.device)
    norm_factor = torch.empty((seq_len,), device=x.device, dtype=_kernel_dtype(x.device))
    out = torch.empty((seq_len, hidden), device=x.device, dtype=_kernel_dtype(x.device))
    _PI0_INFER.rmsnorm_factor_kernel[(max(1, min(128, seq_len)),)](
        x,
        norm_factor,
        seq_len,
        features,
        eps=1e-6,
        BLOCK_SIZE=max(128, _head_block_size(features) * 32),
    )
    block_n = _default_block_size(seq_len)
    block_m = _default_block_size(hidden)
    block_k = _default_block_size(features)
    _PI0_INFER.scaled_matmul_small_gate[_matmul_grid(seq_len, hidden, block_n=block_n, block_m=block_m)](
        x,
        norm_factor,
        weight1,
        weight2,
        out,
        seq_len=seq_len,
        features=features,
        hidden=hidden,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
    )
    return out


def _rms_matmul_bias_kernel(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    seq_len, features = map(int, x.shape)
    hidden = int(weight.shape[1])
    x = _to_kernel_tensor(x, device=x.device)
    weight = _to_kernel_tensor(weight, device=x.device)
    bias = _to_kernel_tensor(bias, device=x.device)
    norm_factor = torch.empty((seq_len,), device=x.device, dtype=_kernel_dtype(x.device))
    out = torch.zeros((seq_len, hidden), device=x.device, dtype=_kernel_dtype(x.device))
    _PI0_INFER.rmsnorm_factor_kernel[(max(1, min(128, seq_len)),)](
        x,
        norm_factor,
        seq_len,
        features,
        eps=1e-6,
        BLOCK_SIZE=max(128, _head_block_size(features) * 32),
    )
    block_n = _default_block_size(seq_len)
    block_m = _default_block_size(hidden)
    block_k = _default_block_size(features)
    _PI0_INFER.scaled_matmul_small_bias_res[_matmul_grid(seq_len, hidden, block_n=block_n, block_m=block_m)](
        x,
        norm_factor,
        weight,
        out,
        bias,
        out,
        seq_len=seq_len,
        features=features,
        hidden=hidden,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
    )
    return out


def _rms_qkv_rope_kernel(
    x: torch.Tensor,
    weight_qkv: torch.Tensor,
    rope_weights: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    safe_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len, features = map(int, x.shape)
    x = _to_kernel_tensor(x, device=x.device)
    weight_qkv = _to_kernel_tensor(weight_qkv, device=x.device)
    rope_weights = _to_kernel_tensor(rope_weights, device=x.device)
    identity_rope = _identity_pairwise_rope_weights(rows=seq_len, head_dim=int(head_dim), device=x.device)
    norm_factor = torch.empty((seq_len,), device=x.device, dtype=torch.float32)
    q = torch.empty((seq_len, int(num_heads) * int(head_dim)), device=x.device, dtype=_kernel_dtype(x.device))
    k = torch.empty((seq_len, int(head_dim)), device=x.device, dtype=_kernel_dtype(x.device))
    v = torch.empty((seq_len, int(head_dim)), device=x.device, dtype=_kernel_dtype(x.device))
    _PI0_INFER.rmsnorm_factor_kernel[(max(1, min(128, seq_len)),)](
        x,
        norm_factor,
        seq_len,
        features,
        eps=1e-6,
        BLOCK_SIZE=max(128, _head_block_size(features) * 32),
    )
    if int(head_dim) >= 16 and not bool(safe_kernel):
        head_block = max(16, _head_block_size(head_dim))
        grid = max(
            1,
            min(128, triton.cdiv(seq_len, 32) * triton.cdiv((int(num_heads) + 2) * int(head_dim), head_block)),
        )
        _PI0_INFER.scaled_matmul_rope_qkv[(grid,)](
            x,
            norm_factor,
            seq_len,
            features,
            int(head_dim),
            int(num_heads),
            weight_qkv,
            identity_rope,
            q,
            k,
            v,
            BLOCK_SIZE_M=32 if seq_len < 32 else 64,
            BLOCK_SIZE_N=head_block,
            BLOCK_SIZE_K=32 if features < 64 else 64,
        )
    else:
        grid = max(
            1,
            min(128, triton.cdiv(seq_len, 16) * triton.cdiv((int(num_heads) + 2) * int(head_dim), 16)),
        )
        _scaled_matmul_rope_qkv_small_kernel[(grid,)](
            x,
            norm_factor,
            weight_qkv,
            identity_rope,
            q,
            k,
            v,
            seq_len=seq_len,
            features=features,
            head_dim=int(head_dim),
            num_heads=int(num_heads),
            BLOCK_ROWS=16,
            BLOCK_COLS=16,
            BLOCK_K=32 if features < 64 else 64,
        )
    return (
        _apply_gemma_rope_kernel(q, rope_weights, head_dim=int(head_dim)),
        _apply_gemma_rope_kernel(k, rope_weights, head_dim=int(head_dim)),
        v,
    )


def _layer_norm_matmul_bias_kernel(
    x: torch.Tensor,
    norm_w: torch.Tensor,
    norm_b: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    seq_len, features = map(int, x.shape)
    hidden = int(weight.shape[1])
    x = _to_kernel_tensor(x, device=x.device)
    norm_w = _to_kernel_tensor(norm_w, device=x.device)
    norm_b = _to_kernel_tensor(norm_b, device=x.device)
    weight = _to_kernel_tensor(weight, device=x.device)
    bias = _to_kernel_tensor(bias, device=x.device)
    x_norm = torch.empty_like(x)
    out = torch.empty((seq_len, hidden), device=x.device, dtype=_kernel_dtype(x.device))
    _PI0_INFER.layer_norm_small_kernel[(seq_len,)](
        x,
        x_norm,
        norm_w,
        norm_b,
        seq_len=seq_len,
        features=features,
        eps=1e-5,
    )
    block_n = _default_block_size(seq_len)
    block_m = _default_block_size(hidden)
    block_k = _default_block_size(features)
    _PI0_INFER.matmul_small_bias[_matmul_grid(seq_len, hidden, block_n=block_n, block_m=block_m)](
        x_norm,
        weight,
        out,
        bias,
        seq_len=seq_len,
        features=features,
        hidden=hidden,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
    )
    return out


@triton.jit
def _apply_gemma_rope_triton_kernel(
    inp_ptr,
    rope_ptr,
    out_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_ROWS: tl.constexpr = 8,
    BLOCK_COLS: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_rows = tl.cdiv(rows, BLOCK_ROWS)
    grid_cols = tl.cdiv(cols, BLOCK_COLS)
    half_dim = head_dim // 2
    while pid < grid_rows * grid_cols:
        pid_row = pid // grid_cols
        pid_col = pid % grid_cols
        offs_row = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        offs_col = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)[None, :]
        local_col = offs_col % head_dim
        head_base = offs_col - local_col
        pair_local = tl.where(local_col < half_dim, local_col + half_dim, local_col - half_dim)
        pair_col = head_base + pair_local
        rope_col = tl.where(local_col < half_dim, local_col, local_col - half_dim)

        x = tl.load(
            inp_ptr + offs_row * cols + offs_col,
            mask=(offs_row < rows) & (offs_col < cols),
            other=0.0,
        ).to(tl.float32)
        pair = tl.load(
            inp_ptr + offs_row * cols + pair_col,
            mask=(offs_row < rows) & (pair_col < cols),
            other=0.0,
        ).to(tl.float32)
        cos = tl.load(
            rope_ptr + offs_row * head_dim + rope_col,
            mask=(offs_row < rows) & (rope_col < half_dim),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            rope_ptr + offs_row * head_dim + (rope_col + half_dim),
            mask=(offs_row < rows) & (rope_col < half_dim),
            other=0.0,
        ).to(tl.float32)

        out = tl.where(local_col < half_dim, x * cos - pair * sin, x * cos + pair * sin)
        tl.store(
            out_ptr + offs_row * cols + offs_col,
            out.to(tl.bfloat16),
            mask=(offs_row < rows) & (offs_col < cols),
        )
        pid += psize


def _apply_gemma_rope_kernel(x: torch.Tensor, rope_weights: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    rows, cols = map(int, x.shape)
    x = _to_kernel_tensor(x, device=x.device)
    rope_weights = _to_kernel_tensor(rope_weights, device=x.device)
    out = torch.empty_like(x)
    _apply_gemma_rope_triton_kernel[(max(1, min(128, _matmul_grid(rows, cols, block_n=32, block_m=32)[0])),)](
        x,
        rope_weights,
        out,
        rows=rows,
        cols=cols,
        head_dim=int(head_dim),
        BLOCK_ROWS=8,
        BLOCK_COLS=32 if cols >= 32 else 16,
    )
    return out


def _identity_pairwise_rope_weights(*, rows: int, head_dim: int, device: torch.device) -> torch.Tensor:
    rope = torch.zeros((int(rows), int(head_dim)), device=device, dtype=_kernel_dtype(device))
    rope[:, 0::2] = 1
    return rope.contiguous()


@triton.jit
def _scaled_matmul_rope_qkv_small_kernel(
    inp_ptr,
    inp_norm_factor_ptr,
    weight_qkv_ptr,
    rope_weights_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    BLOCK_ROWS: tl.constexpr = 16,
    BLOCK_COLS: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    total_out = (num_heads + 2) * head_dim
    grid_rows = tl.cdiv(seq_len, BLOCK_ROWS)
    grid_cols = tl.cdiv(total_out, BLOCK_COLS)
    while pid < grid_rows * grid_cols:
        pid_row = pid // grid_cols
        pid_col = pid % grid_cols
        start_i = pid_row * BLOCK_ROWS
        start_j = pid_col * BLOCK_COLS
        offs_i = start_i + tl.arange(0, BLOCK_ROWS)[:, None]
        offs_j = start_j + tl.arange(0, BLOCK_COLS)[None, :]
        norm_factor = tl.load(
            inp_norm_factor_ptr + start_i + tl.arange(0, BLOCK_ROWS),
            mask=start_i + tl.arange(0, BLOCK_ROWS) < seq_len,
            other=0.0,
        ).to(tl.float32)
        acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)
        for k0 in range(0, features, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(
                inp_ptr + offs_i * features + offs_k[None, :],
                mask=(offs_i < seq_len) & (offs_k[None, :] < features),
                other=0.0,
            ).to(tl.float32)
            x = x * norm_factor[:, None]
            w = tl.load(
                weight_qkv_ptr + offs_k[:, None] * total_out + offs_j,
                mask=(offs_k[:, None] < features) & (offs_j < total_out),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(x[:, :, None] * w[None, :, :], axis=1)
        qk_cols = (num_heads + 1) * head_dim
        if start_j < qk_cols:
            x0, x1 = tl.split(acc.reshape(BLOCK_ROWS, BLOCK_COLS // 2, 2))
            rope = tl.load(
                rope_weights_ptr + offs_i * head_dim + offs_j % head_dim,
                mask=(offs_i < seq_len) & (offs_j < qk_cols),
                other=0.0,
            ).to(tl.float32)
            rope_cos, rope_sin = tl.split(rope.reshape(BLOCK_ROWS, BLOCK_COLS // 2, 2))
            x0_out = x0 * rope_cos - x1 * rope_sin
            x1_out = x1 * rope_cos + x0 * rope_sin
            rotated = tl.interleave(x0_out, x1_out)
            acc = tl.where(offs_j < qk_cols, rotated, acc)
        acc = acc.to(tl.bfloat16)
        q_mask = (offs_i < seq_len) & (offs_j < num_heads * head_dim)
        tl.store(q_ptr + offs_i * (num_heads * head_dim) + offs_j, acc, mask=q_mask)
        k_mask = (offs_i < seq_len) & (offs_j >= num_heads * head_dim) & (offs_j < (num_heads + 1) * head_dim)
        tl.store(k_ptr + offs_i * head_dim + (offs_j - num_heads * head_dim), acc, mask=k_mask)
        v_mask = (offs_i < seq_len) & (offs_j >= (num_heads + 1) * head_dim) & (offs_j < total_out)
        tl.store(v_ptr + offs_i * head_dim + (offs_j - (num_heads + 1) * head_dim), acc, mask=v_mask)
        pid += psize


@triton.jit
def _mix_with_timestep_kernel(
    noise_ptr,
    x0_ptr,
    timestep_ptr,
    out_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    horizon: tl.constexpr,
    BLOCK_ROWS: tl.constexpr = 16,
    BLOCK_COLS: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_rows = tl.cdiv(rows, BLOCK_ROWS)
    grid_cols = tl.cdiv(cols, BLOCK_COLS)
    while pid < grid_rows * grid_cols:
        pid_row = pid // grid_cols
        pid_col = pid % grid_cols
        offs_row = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offs_col = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        batch_idx = offs_row // horizon
        t = tl.load(timestep_ptr + batch_idx, mask=offs_row < rows, other=0.0).to(tl.float32)
        noise = tl.load(
            noise_ptr + offs_row[:, None] * cols + offs_col[None, :],
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < cols),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(
            x0_ptr + offs_row[:, None] * cols + offs_col[None, :],
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < cols),
            other=0.0,
        ).to(tl.float32)
        out = t[:, None] * noise + (1.0 - t[:, None]) * x0
        tl.store(
            out_ptr + offs_row[:, None] * cols + offs_col[None, :],
            out.to(tl.bfloat16),
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < cols),
        )
        pid += psize


@triton.jit
def _x0_hat_kernel(
    x_t_ptr,
    velocity_ptr,
    timestep_ptr,
    out_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    horizon: tl.constexpr,
    BLOCK_ROWS: tl.constexpr = 16,
    BLOCK_COLS: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_rows = tl.cdiv(rows, BLOCK_ROWS)
    grid_cols = tl.cdiv(cols, BLOCK_COLS)
    while pid < grid_rows * grid_cols:
        pid_row = pid // grid_cols
        pid_col = pid % grid_cols
        offs_row = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offs_col = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        batch_idx = offs_row // horizon
        t = tl.load(timestep_ptr + batch_idx, mask=offs_row < rows, other=0.0).to(tl.float32)
        x_t = tl.load(
            x_t_ptr + offs_row[:, None] * cols + offs_col[None, :],
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < cols),
            other=0.0,
        ).to(tl.float32)
        velocity = tl.load(
            velocity_ptr + offs_row[:, None] * cols + offs_col[None, :],
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < cols),
            other=0.0,
        ).to(tl.float32)
        out = x_t - t[:, None] * velocity
        tl.store(
            out_ptr + offs_row[:, None] * cols + offs_col[None, :],
            out.to(tl.bfloat16),
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < cols),
        )
        pid += psize


@triton.jit
def _spec_verify_accept_metrics_kernel(
    x0_hat_ptr,
    x0_draft_ptr,
    gripper_prev_ptr,
    accepted_ptr,
    action_prefix_ptr,
    stop_mask_ptr,
    cut_mask_ptr,
    metrics_ptr,
    batch_size: tl.constexpr,
    verify_k: tl.constexpr,
    horizon: tl.constexpr,
    action_dim: tl.constexpr,
    eval_h: tl.constexpr,
    eval_d: tl.constexpr,
    tau_radius: tl.constexpr,
    gripper_switch_threshold: tl.constexpr,
    has_gripper_prev: tl.constexpr,
    enable_gripper_verify: tl.constexpr,
    enable_gripper_post_verify: tl.constexpr,
):
    dist_sum = tl.full((), 0.0, tl.float32)
    accepted_sum = tl.full((), 0.0, tl.float32)
    stop_sum = tl.full((), 0.0, tl.float32)
    cut_sum = tl.full((), 0.0, tl.float32)
    scheduled = tl.full((), False, tl.int1)
    norm = tl.sqrt(tl.full((), eval_d, tl.float32))

    for b in range(0, batch_size):
        accepted_base = tl.full((), 0, tl.int64)
        prefix_active = tl.full((), True, tl.int1)

        for h in range(0, eval_h):
            step_ok = tl.full((), True, tl.int1)
            for k in range(0, verify_k):
                ss = tl.full((), 0.0, tl.float32)
                for d in range(0, eval_d):
                    hat = tl.load(x0_hat_ptr + ((b * verify_k + k) * horizon + h) * action_dim + d).to(tl.float32)
                    draft = tl.load(x0_draft_ptr + (b * horizon + h) * action_dim + d).to(tl.float32)
                    diff = hat - draft
                    ss += diff * diff
                dist = tl.sqrt(ss) / norm
                dist_sum += dist
                step_ok = step_ok & (dist <= tau_radius)
            accepted_base = tl.where(prefix_active & step_ok, accepted_base + 1, accepted_base)
            prefix_active = prefix_active & step_ok

        verify_stop = tl.full((), False, tl.int1)
        if enable_gripper_verify:
            if has_gripper_prev:
                if action_dim >= 7:
                    prev0 = tl.load(gripper_prev_ptr + b).to(tl.float32)
                    for k in range(0, verify_k):
                        for h in range(0, eval_h):
                            prev_value = prev0
                            if h > 0:
                                prev_value = tl.load(
                                    x0_hat_ptr + ((b * verify_k + k) * horizon + (h - 1)) * action_dim + 6
                                ).to(tl.float32)
                            curr_value = tl.load(x0_hat_ptr + ((b * verify_k + k) * horizon + h) * action_dim + 6).to(
                                tl.float32
                            )
                            crossed = (
                                ((prev_value < gripper_switch_threshold) & (curr_value >= gripper_switch_threshold))
                                | ((prev_value >= gripper_switch_threshold) & (curr_value < gripper_switch_threshold))
                            )
                            verify_stop = verify_stop | crossed

        action_prefix = tl.where(verify_stop, 0, accepted_base)
        accepted_final = action_prefix
        cut = tl.full((), False, tl.int1)
        first_cut = tl.full((), 0, tl.int64)
        if enable_gripper_post_verify:
            if has_gripper_prev:
                if action_dim >= 7:
                    prev0 = tl.load(gripper_prev_ptr + b).to(tl.float32)
                    for h in range(0, eval_h):
                        prev_value = prev0
                        if h > 0:
                            prev_value = tl.load(x0_draft_ptr + (b * horizon + (h - 1)) * action_dim + 6).to(
                                tl.float32
                            )
                        curr_value = tl.load(x0_draft_ptr + (b * horizon + h) * action_dim + 6).to(tl.float32)
                        active = h < action_prefix
                        crossed = active & (
                            ((prev_value < gripper_switch_threshold) & (curr_value >= gripper_switch_threshold))
                            | ((prev_value >= gripper_switch_threshold) & (curr_value < gripper_switch_threshold))
                        )
                        new_cut = (~cut) & crossed
                        first_cut = tl.where(new_cut, h, first_cut)
                        cut = cut | crossed
                    accepted_final = tl.where(cut, first_cut, accepted_final)

        cut = cut & (~verify_stop)
        tl.store(accepted_ptr + b, accepted_final)
        tl.store(action_prefix_ptr + b, action_prefix)
        tl.store(stop_mask_ptr + b, verify_stop)
        tl.store(cut_mask_ptr + b, cut)
        accepted_sum += accepted_final.to(tl.float32)
        stop_sum += tl.where(verify_stop, 1.0, 0.0)
        cut_sum += tl.where(cut, 1.0, 0.0)
        scheduled = scheduled | verify_stop | cut

    denom = tl.full((), batch_size * verify_k * eval_h, tl.float32)
    batch_denom = tl.full((), batch_size, tl.float32)
    tl.store(metrics_ptr + 0, dist_sum / denom)
    tl.store(metrics_ptr + 1, accepted_sum / batch_denom)
    tl.store(metrics_ptr + 2, cut_sum / batch_denom)
    tl.store(metrics_ptr + 3, tl.where(scheduled, 1.0, 0.0))
    tl.store(metrics_ptr + 4, stop_sum / batch_denom)


@triton.jit
def _spec_verify_stitch_kernel(
    x0_hat_ptr,
    x0_draft_ptr,
    out_ptr,
    action_prefix_ptr,
    stop_mask_ptr,
    total_values: tl.constexpr,
    verify_k: tl.constexpr,
    horizon: tl.constexpr,
    action_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 256,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    while pid * BLOCK_SIZE < total_values:
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_values
        d = offs % action_dim
        h = (offs // action_dim) % horizon
        b = offs // (horizon * action_dim)
        tail = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
        for k in range(0, verify_k):
            tail += tl.load(
                x0_hat_ptr + ((b * verify_k + k) * horizon + h) * action_dim + d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        tail = tail / tl.full((), verify_k, tl.float32)
        draft = tl.load(x0_draft_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        action_prefix = tl.load(action_prefix_ptr + b, mask=mask, other=0)
        verify_stop = tl.load(stop_mask_ptr + b, mask=mask, other=1).to(tl.int1)
        use_draft = (h < action_prefix) & (~verify_stop)
        out = tl.where(use_draft, draft, tail)
        tl.store(out_ptr + offs, out, mask=mask)
        pid += psize


@triton.jit
def _interpolate_time_bias_kernel(
    time_bias_ptr,
    timestep_ptr,
    out_ptr,
    rows: tl.constexpr,
    steps: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_ROWS: tl.constexpr = 8,
    BLOCK_COLS: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_rows = tl.cdiv(rows, BLOCK_ROWS)
    grid_cols = tl.cdiv(hidden, BLOCK_COLS)
    while pid < grid_rows * grid_cols:
        pid_row = pid // grid_cols
        pid_col = pid % grid_cols
        offs_row = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offs_col = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        timestep = tl.load(timestep_ptr + offs_row, mask=offs_row < rows, other=0.0).to(tl.float32)
        pos = (1.0 - timestep) * steps
        pos = tl.maximum(0.0, tl.minimum(pos, float(steps - 1)))
        lo = pos.to(tl.int32)
        hi = tl.minimum(lo + 1, steps - 1)
        alpha = pos - lo.to(tl.float32)
        lo_bias = tl.load(
            time_bias_ptr + lo[:, None] * hidden + offs_col[None, :],
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        hi_bias = tl.load(
            time_bias_ptr + hi[:, None] * hidden + offs_col[None, :],
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        out = lo_bias + (hi_bias - lo_bias) * alpha[:, None]
        tl.store(
            out_ptr + offs_row[:, None] * hidden + offs_col[None, :],
            out.to(tl.bfloat16),
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < hidden),
        )
        pid += psize


@triton.jit
def _add_rowwise_bias_silu_kernel(
    inp_ptr,
    bias_ptr,
    out_ptr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    BLOCK_ROWS: tl.constexpr = 8,
    BLOCK_COLS: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_rows = tl.cdiv(rows, BLOCK_ROWS)
    grid_cols = tl.cdiv(hidden, BLOCK_COLS)
    while pid < grid_rows * grid_cols:
        pid_row = pid // grid_cols
        pid_col = pid % grid_cols
        offs_row = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offs_col = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        x = tl.load(
            inp_ptr + offs_row[:, None] * hidden + offs_col[None, :],
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + offs_row[:, None] * hidden + offs_col[None, :],
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < hidden),
            other=0.0,
        ).to(tl.float32)
        out = x + bias
        out = out * tl.sigmoid(out)
        tl.store(
            out_ptr + offs_row[:, None] * hidden + offs_col[None, :],
            out.to(tl.bfloat16),
            mask=(offs_row[:, None] < rows) & (offs_col[None, :] < hidden),
        )
        pid += psize


@triton.jit
def _attention_logits_kernel(
    q_ptr,
    prefix_k_ptr,
    suffix_k_ptr,
    out_ptr,
    total_rows: tl.constexpr,
    total_keys: tl.constexpr,
    prefix_len: tl.constexpr,
    suffix_len: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    query_len: tl.constexpr,
    BLOCK_ROWS: tl.constexpr = 4,
    BLOCK_KEYS: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_rows = tl.cdiv(total_rows, BLOCK_ROWS)
    grid_cols = tl.cdiv(total_keys, BLOCK_KEYS)
    scale = 1.0 / tl.sqrt(tl.full([], head_dim, dtype=tl.float32))
    while pid < grid_rows * grid_cols:
        pid_row = pid // grid_cols
        pid_col = pid % grid_cols
        offs_row = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offs_key = pid_col * BLOCK_KEYS + tl.arange(0, BLOCK_KEYS)
        sample_idx = offs_row // (num_heads * query_len)
        acc = tl.zeros((BLOCK_ROWS, BLOCK_KEYS), dtype=tl.float32)
        for k0 in range(0, head_dim, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            q = tl.load(
                q_ptr + offs_row[:, None] * head_dim + offs_k[None, :],
                mask=(offs_row[:, None] < total_rows) & (offs_k[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            prefix_vals = tl.load(
                prefix_k_ptr
                + ((sample_idx[:, None, None] * prefix_len + offs_key[None, :, None]) * head_dim + offs_k[None, None, :]),
                mask=(
                    (offs_row[:, None, None] < total_rows)
                    & (offs_key[None, :, None] < prefix_len)
                    & (offs_k[None, None, :] < head_dim)
                ),
                other=0.0,
            ).to(tl.float32)
            suffix_vals = tl.load(
                suffix_k_ptr
                + (
                    (sample_idx[:, None, None] * suffix_len + (offs_key[None, :, None] - prefix_len)) * head_dim
                    + offs_k[None, None, :]
                ),
                mask=(
                    (offs_row[:, None, None] < total_rows)
                    & (offs_key[None, :, None] >= prefix_len)
                    & (offs_key[None, :, None] < total_keys)
                    & (offs_k[None, None, :] < head_dim)
                ),
                other=0.0,
            ).to(tl.float32)
            k = tl.where(offs_key[None, :, None] < prefix_len, prefix_vals, suffix_vals)
            acc += tl.sum(q[:, None, :] * k, axis=2)
        tl.store(
            out_ptr + offs_row[:, None] * total_keys + offs_key[None, :],
            (acc * scale).to(tl.float32),
            mask=(offs_row[:, None] < total_rows) & (offs_key[None, :] < total_keys),
        )
        pid += psize


@triton.jit
def _attention_softmax_mask_kernel(
    inp_ptr,
    key_valid_ptr,
    out_ptr,
    total_rows: tl.constexpr,
    total_keys: tl.constexpr,
    num_heads: tl.constexpr,
    query_len: tl.constexpr,
    prefix_len: tl.constexpr,
    block_suffix_for_query0: tl.constexpr,
    BLOCK_ROWS: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    while pid * BLOCK_ROWS < total_rows:
        offs_row = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        offs_key = tl.arange(0, BLOCK_SIZE)[None, :]
        sample_idx = offs_row // (num_heads * query_len)
        query_idx = offs_row % query_len
        valid = (offs_row < total_rows) & (offs_key < total_keys)
        key_valid = tl.load(
            key_valid_ptr + sample_idx * total_keys + offs_key,
            mask=valid,
            other=0,
        ).to(tl.int1)
        valid = valid & key_valid
        if block_suffix_for_query0:
            valid = valid & ((query_idx != 0) | (offs_key <= prefix_len))
        vals = tl.load(inp_ptr + offs_row * total_keys + offs_key, mask=valid, other=-float("inf"))
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        denom = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        probs = vals / denom
        tl.store(
            out_ptr + offs_row * total_keys + offs_key,
            probs.to(tl.float32),
            mask=(offs_row < total_rows) & (offs_key < total_keys),
        )
        pid += psize


@triton.jit
def _attention_values_kernel(
    prob_ptr,
    prefix_v_ptr,
    suffix_v_ptr,
    out_ptr,
    total_rows: tl.constexpr,
    total_keys: tl.constexpr,
    prefix_len: tl.constexpr,
    suffix_len: tl.constexpr,
    head_dim: tl.constexpr,
    num_heads: tl.constexpr,
    query_len: tl.constexpr,
    BLOCK_ROWS: tl.constexpr = 4,
    BLOCK_KEYS: tl.constexpr = 32,
    BLOCK_COLS: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_rows = tl.cdiv(total_rows, BLOCK_ROWS)
    grid_cols = tl.cdiv(head_dim, BLOCK_COLS)
    while pid < grid_rows * grid_cols:
        pid_row = pid // grid_cols
        pid_col = pid % grid_cols
        offs_row = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offs_col = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        sample_idx = offs_row // (num_heads * query_len)
        acc = tl.zeros((BLOCK_ROWS, BLOCK_COLS), dtype=tl.float32)
        for key0 in range(0, total_keys, BLOCK_KEYS):
            offs_key = key0 + tl.arange(0, BLOCK_KEYS)
            probs = tl.load(
                prob_ptr + offs_row[:, None] * total_keys + offs_key[None, :],
                mask=(offs_row[:, None] < total_rows) & (offs_key[None, :] < total_keys),
                other=0.0,
            ).to(tl.float32)
            prefix_vals = tl.load(
                prefix_v_ptr
                + ((sample_idx[:, None, None] * prefix_len + offs_key[None, :, None]) * head_dim + offs_col[None, None, :]),
                mask=(
                    (offs_row[:, None, None] < total_rows)
                    & (offs_key[None, :, None] < prefix_len)
                    & (offs_col[None, None, :] < head_dim)
                ),
                other=0.0,
            ).to(tl.float32)
            suffix_vals = tl.load(
                suffix_v_ptr
                + (
                    (sample_idx[:, None, None] * suffix_len + (offs_key[None, :, None] - prefix_len)) * head_dim
                    + offs_col[None, None, :]
                ),
                mask=(
                    (offs_row[:, None, None] < total_rows)
                    & (offs_key[None, :, None] >= prefix_len)
                    & (offs_key[None, :, None] < total_keys)
                    & (offs_col[None, None, :] < head_dim)
                ),
                other=0.0,
            ).to(tl.float32)
            values = tl.where(offs_key[None, :, None] < prefix_len, prefix_vals, suffix_vals)
            acc += tl.sum(probs[:, :, None] * values, axis=1)
        tl.store(
            out_ptr + offs_row[:, None] * head_dim + offs_col[None, :],
            acc.to(tl.bfloat16),
            mask=(offs_row[:, None] < total_rows) & (offs_col[None, :] < head_dim),
        )
        pid += psize


@triton.jit
def _pack_grouped_kv_kernel(
    prefix_ptr,
    suffix_ptr,
    out_ptr,
    num_groups: tl.constexpr,
    prefix_len: tl.constexpr,
    suffix_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_ROWS: tl.constexpr = 8,
    BLOCK_COLS: tl.constexpr = 32,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    keys_per_group: tl.constexpr = prefix_len + suffix_len
    total_rows: tl.constexpr = num_groups * keys_per_group
    grid_rows = tl.cdiv(total_rows, BLOCK_ROWS)
    grid_cols = tl.cdiv(head_dim, BLOCK_COLS)
    while pid < grid_rows * grid_cols:
        pid_row = pid // grid_cols
        pid_col = pid % grid_cols
        offs_row = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        offs_col = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        group_idx = offs_row // keys_per_group
        row_in_group = offs_row % keys_per_group
        use_prefix = row_in_group < prefix_len
        prefix_vals = tl.load(
            prefix_ptr + row_in_group[:, None] * head_dim + offs_col[None, :],
            mask=(offs_row[:, None] < total_rows) & use_prefix[:, None] & (offs_col[None, :] < head_dim),
            other=0.0,
        )
        suffix_row = group_idx * suffix_len + tl.maximum(row_in_group - prefix_len, 0)
        suffix_vals = tl.load(
            suffix_ptr + suffix_row[:, None] * head_dim + offs_col[None, :],
            mask=(offs_row[:, None] < total_rows) & (~use_prefix)[:, None] & (offs_col[None, :] < head_dim),
            other=0.0,
        )
        vals = tl.where(use_prefix[:, None], prefix_vals, suffix_vals)
        tl.store(
            out_ptr + offs_row[:, None] * head_dim + offs_col[None, :],
            vals,
            mask=(offs_row[:, None] < total_rows) & (offs_col[None, :] < head_dim),
        )
        pid += psize


@triton.jit
def _grouped_softmax_mask0_kernel(
    inp_ptr,
    out_ptr,
    total_queries: tl.constexpr,
    keys_per_group: tl.constexpr,
    num_groups: tl.constexpr,
    num_heads: tl.constexpr,
    query_len: tl.constexpr,
    prefix_len: tl.constexpr,
    block_suffix_for_query0: tl.constexpr,
    BLOCK_ROWS: tl.constexpr = 4,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    total_keys: tl.constexpr = keys_per_group * num_groups
    rows_per_group: tl.constexpr = query_len * num_heads
    big_neg = -2.3819763e38

    assert BLOCK_SIZE >= total_keys, f"BLOCK_SIZE must be >= total_keys, got {BLOCK_SIZE} < {total_keys}"

    for i in range(pid * BLOCK_ROWS, total_queries, psize * BLOCK_ROWS):
        offs_i = i + tl.arange(0, BLOCK_ROWS)[:, None]
        offs_j = tl.arange(0, BLOCK_SIZE)[None, :]

        group_idx = offs_i // rows_per_group
        row_in_group = offs_i % rows_per_group
        query_idx = row_in_group // num_heads

        key_group = offs_j // keys_per_group
        key_in_group = offs_j % keys_per_group

        valid = (offs_i < total_queries) & (offs_j < total_keys) & (key_group == group_idx)
        if block_suffix_for_query0:
            valid = valid & ((query_idx != 0) | (key_in_group <= prefix_len))

        vals = tl.load(inp_ptr + offs_i * total_keys + offs_j, mask=valid, other=big_neg)
        vals = tl.exp(vals - tl.max(vals, axis=1, keep_dims=True))
        denom = tl.sum(vals, axis=1, keep_dims=True, dtype=tl.float32)
        probs = vals / denom
        tl.store(
            out_ptr + offs_i * total_keys + offs_j,
            probs.to(tl.bfloat16),
            mask=(offs_i < total_queries) & (offs_j < total_keys),
        )


@dataclasses.dataclass(frozen=True)
class PrefixContext:
    prefix_embs: torch.Tensor
    prefix_pad_masks: torch.Tensor
    prefix_att_masks: torch.Tensor


@dataclasses.dataclass(frozen=True)
class FullCacheSnapshot:
    encoder_seq_len: int
    encoder_x: torch.Tensor | None = None
    encoder_k: torch.Tensor | None = None
    encoder_v: torch.Tensor | None = None


@dataclasses.dataclass(frozen=True)
class SpecVerifyOutput:
    x0_hat: torch.Tensor
    actions: torch.Tensor
    metrics: torch.Tensor
    accepted_prefix_len: torch.Tensor


@dataclasses.dataclass
class VerifyFastBuffers:
    decoder_x: torch.Tensor
    decoder_x_buf: torch.Tensor
    decoder_state_buf: torch.Tensor
    decoder_norm_factor_buf: torch.Tensor
    decoder_q_buf: torch.Tensor
    decoder_attn_buf: torch.Tensor
    decoder_hidden: torch.Tensor
    grouped_k_buf: torch.Tensor
    grouped_v_buf: torch.Tensor
    suffix_k_buf: torch.Tensor
    suffix_v_buf: torch.Tensor
    velocity_buf: torch.Tensor


@dataclasses.dataclass
class VerifyPostprocessBuffers:
    accepted_prefix_len: torch.Tensor
    action_prefix_len: torch.Tensor
    gripper_verify_stop_mask: torch.Tensor
    gripper_switch_cut_mask: torch.Tensor
    metrics: torch.Tensor
    actions: torch.Tensor


@dataclasses.dataclass(frozen=True)
class _VerifyGraphRuntime:
    weights: Mapping[str, torch.Tensor]


@dataclasses.dataclass
class _VerifyFastGraph:
    key: tuple[Any, ...]
    graph: Any
    x_t_bk: torch.Tensor
    timestep_bk: torch.Tensor
    state: torch.Tensor
    encoder_k: torch.Tensor
    encoder_v: torch.Tensor
    cache_snapshot: FullCacheSnapshot
    full_runtime: _VerifyGraphRuntime
    buffers: VerifyFastBuffers
    output: torch.Tensor
    encoder_k_source: tuple[int, int] | None = None
    encoder_v_source: tuple[int, int] | None = None


@dataclasses.dataclass
class _VerifySemanticsGraph:
    key: tuple[Any, ...]
    graph: Any
    noise: torch.Tensor
    x0_draft: torch.Tensor
    state: torch.Tensor
    last_gripper: torch.Tensor | None
    encoder_k: torch.Tensor
    encoder_v: torch.Tensor
    cache_snapshot: FullCacheSnapshot
    full_runtime: _VerifyGraphRuntime
    output: SpecVerifyOutput
    encoder_k_source: tuple[int, int] | None = None
    encoder_v_source: tuple[int, int] | None = None


@dataclasses.dataclass
class _DraftGraph:
    key: tuple[Any, ...]
    graph: Any
    prefix_embs: torch.Tensor
    prefix_pad_masks: torch.Tensor
    prefix_att_masks: torch.Tensor
    state: torch.Tensor
    prefix: PrefixContext
    output: torch.Tensor


class Pi0SpecInference:
    def __init__(
        self,
        *,
        checkpoint: Mapping[str, torch.Tensor],
        draft_checkpoint: Mapping[str, Any],
        num_views: int,
        chunk_size: int,
    ) -> None:
        language_embeds = torch.as_tensor(checkpoint["language_embeds"])
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size)
        self.prompt_len = int(language_embeds.shape[0])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.weights: dict[str, torch.Tensor] = {
            "language_embeds": language_embeds.to(device=self.device, dtype=self.compute_dtype).contiguous(),
        }

        self.meta = dict(draft_checkpoint.get("meta", {}) or {})
        self.hidden_size = int(self.meta.get("img_dim", language_embeds.shape[-1]))
        self.chunk_m = min(int(self.meta.get("chunk_m", self.chunk_size)), self.chunk_size)
        self.out_dim = int(self.meta.get("out_dim", 7))
        self.num_heads = int(self.meta.get("draft_num_heads", max(1, self.hidden_size // 256)))
        self.num_kv_heads = int(self.meta.get("draft_num_kv_heads", 1))
        self.head_dim = int(self.meta.get("draft_head_dim", max(1, self.hidden_size // max(1, self.num_heads))))
        if int(self.num_kv_heads) != 1:
            raise ValueError(f"Spec Triton draft currently requires num_kv_heads=1, got {self.num_kv_heads}")

        raw_draft = {
            str(k): v.to(device=self.device)
            for k, v in draft_checkpoint.items()
            if str(k) != "meta" and isinstance(v, torch.Tensor)
        }
        self._draft = {
            "draft_state_in_proj_w": _canonicalize_linear_weight(
                raw_draft["draft_state_in_proj_w"],
                in_features=32,
                out_features=self.hidden_size,
            ).to(dtype=self.compute_dtype),
            "draft_state_in_proj_b": raw_draft["draft_state_in_proj_b"].to(dtype=self.compute_dtype).contiguous(),
            "draft_action_queries": raw_draft["draft_action_queries"].to(dtype=self.compute_dtype).contiguous(),
            "draft_qkv_w": _canonicalize_linear_weight(
                raw_draft["draft_qkv_w"],
                in_features=self.hidden_size,
                out_features=(self.num_heads + 2) * self.head_dim,
            ).to(dtype=self.compute_dtype),
            # PyTorch Linear stores [out, in]; this layer is square so shape-based
            # canonicalization cannot disambiguate the required transpose.
            "draft_attn_o_w": raw_draft["draft_attn_o_w"].transpose(0, 1).to(dtype=self.compute_dtype).contiguous(),
            "draft_ffn_gate_w": _canonicalize_linear_weight(
                raw_draft["draft_ffn_gate_w"],
                in_features=self.hidden_size,
                out_features=int(raw_draft["draft_ffn_gate_w"].shape[0]),
            ).to(dtype=self.compute_dtype),
            "draft_ffn_up_w": _canonicalize_linear_weight(
                raw_draft["draft_ffn_up_w"],
                in_features=self.hidden_size,
                out_features=int(raw_draft["draft_ffn_up_w"].shape[0]),
            ).to(dtype=self.compute_dtype),
            "draft_ffn_down_w": _canonicalize_linear_weight(
                raw_draft["draft_ffn_down_w"],
                in_features=int(raw_draft["draft_ffn_down_w"].shape[1]),
                out_features=self.hidden_size,
            ).to(dtype=self.compute_dtype),
            "draft_input_layernorm_w": raw_draft["draft_input_layernorm_w"].to(dtype=self.compute_dtype).contiguous(),
            "draft_post_attention_layernorm_w": raw_draft["draft_post_attention_layernorm_w"]
            .to(dtype=self.compute_dtype)
            .contiguous(),
            "draft_action_head_w": _canonicalize_linear_weight(
                raw_draft["draft_action_head_w"],
                in_features=self.hidden_size,
                out_features=self.out_dim,
            ).to(dtype=self.compute_dtype),
            "draft_action_head_b": raw_draft["draft_action_head_b"].to(dtype=self.compute_dtype).contiguous(),
        }
        draft_input_scale = (1.0 + self._draft["draft_input_layernorm_w"]).contiguous()
        draft_post_attn_scale = (1.0 + self._draft["draft_post_attention_layernorm_w"]).contiguous()
        self._draft["draft_qkv_w_scaled"] = (self._draft["draft_qkv_w"] * draft_input_scale[:, None]).contiguous()
        self._draft["draft_ffn_gate_w_scaled"] = (
            self._draft["draft_ffn_gate_w"] * draft_post_attn_scale[:, None]
        ).contiguous()
        self._draft["draft_ffn_up_w_scaled"] = (
            self._draft["draft_ffn_up_w"] * draft_post_attn_scale[:, None]
        ).contiguous()
        self._vision_prefix_buf: torch.Tensor | None = None
        self._vision_prefix_norm_buf: torch.Tensor | None = None
        self._draft_graph: _DraftGraph | None = None
        self._draft_graph_failed = False
        self._verify_fast_buffers: VerifyFastBuffers | None = None
        self._verify_fast_buffers_key: tuple[int, ...] | None = None
        self._verify_postprocess_buffers: VerifyPostprocessBuffers | None = None
        self._verify_postprocess_buffers_key: tuple[Any, ...] | None = None
        self._verify_fast_graph: _VerifyFastGraph | None = None
        self._verify_fast_graph_failed = False
        self._verify_semantics_graph: _VerifySemanticsGraph | None = None
        self._verify_semantics_graph_failed = False
        self._rope_cache: dict[tuple[int, int, int, str], torch.Tensor] = {}
        self._gemma_rope_cache: dict[tuple[int, int, int, str], torch.Tensor] = {}
        self._verify_timestep_cache: dict[tuple[tuple[float, ...], str], torch.Tensor] = {}

    def _project_vision_prefix(
        self,
        *,
        full_runtime: Any,
    ) -> torch.Tensor:
        vision_x = full_runtime.buffers["vision_x"]
        vision_final_norm_w = full_runtime.weights["vision_final_norm_w"]
        vision_final_norm_b = full_runtime.weights["vision_final_norm_b"]
        projector_w = full_runtime.weights["encoder_multi_modal_projector_w"]
        projector_b = full_runtime.weights["encoder_multi_modal_projector_b"]

        num_views, num_tokens, features = map(int, vision_x.shape)
        hidden = int(projector_w.shape[1] if int(projector_w.shape[0]) == features else projector_w.shape[0])

        can_use_triton_projector = (
            vision_x.device.type == "cuda"
            and hasattr(full_runtime, "_infer_module")
            and hasattr(full_runtime._infer_module, "layer_norm_matmul_n256_1152_2048_bias")
            and int(num_tokens) == 256
            and int(features) == 1152
            and int(hidden) == 2048
        )
        if can_use_triton_projector:
            if self._vision_prefix_buf is None or tuple(self._vision_prefix_buf.shape) != (num_views * num_tokens, hidden):
                self._vision_prefix_buf = torch.empty(
                    (num_views * num_tokens, hidden),
                    device=vision_x.device,
                    dtype=projector_b.dtype,
                )
            if self._vision_prefix_norm_buf is None or tuple(self._vision_prefix_norm_buf.shape) != tuple(vision_x.shape):
                self._vision_prefix_norm_buf = torch.empty_like(vision_x)
            full_runtime._infer_module.layer_norm_matmul_n256_1152_2048_bias(
                vision_x,
                vision_final_norm_w,
                vision_final_norm_b,
                projector_w,
                projector_b,
                self._vision_prefix_buf,
                self._vision_prefix_norm_buf,
            )
            return self._vision_prefix_buf.view(1, num_views * num_tokens, hidden).to(dtype=torch.float32)

        projector_w = _canonicalize_linear_weight(
            projector_w,
            in_features=features,
            out_features=hidden,
        )
        vision_proj = _layer_norm_matmul_bias_kernel(
            vision_x.reshape(num_views * num_tokens, features),
            vision_final_norm_w,
            vision_final_norm_b,
            projector_w,
            projector_b,
        )
        return vision_proj.reshape(1, num_views * num_tokens, hidden).to(dtype=torch.float32).contiguous()

    def _rope_weights(
        self,
        *,
        seq_len: int,
        batch_size: int,
        head_dim: int,
        device: torch.device,
        position_offset: int = 0,
    ) -> torch.Tensor:
        cache_key = (int(seq_len), int(head_dim), int(position_offset), str(device))
        rope = self._rope_cache.get(cache_key)
        if rope is None:
            inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, int(head_dim), 2, dtype=torch.float32, device=device) / float(head_dim))
            )
            positions = torch.arange(int(seq_len), device=device, dtype=torch.float32)[:, None] + float(position_offset)
            angles = positions * inv_freq[None, :]
            rope = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).reshape(int(seq_len), int(head_dim))
            rope = rope.to(dtype=self.compute_dtype).contiguous()
            self._rope_cache[cache_key] = rope
        if int(batch_size) == 1:
            return rope
        return rope.repeat(int(batch_size), 1).contiguous()

    def _gemma_rope_weights(
        self,
        *,
        seq_len: int,
        batch_size: int,
        head_dim: int,
        device: torch.device,
        position_offset: int = 0,
    ) -> torch.Tensor:
        cache_key = (int(seq_len), int(head_dim), int(position_offset), str(device))
        rope = self._gemma_rope_cache.get(cache_key)
        if rope is None:
            half_dim = int(head_dim) // 2
            inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, int(head_dim), 2, dtype=torch.float32, device=device) / float(head_dim))
            )
            positions = torch.arange(int(seq_len), device=device, dtype=torch.float32)[:, None] + float(position_offset)
            angles = positions * inv_freq[None, :]
            rope = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
            rope = rope.to(dtype=self.compute_dtype).contiguous()
            if int(rope.shape[1]) != int(head_dim) or int(half_dim) * 2 != int(head_dim):
                raise ValueError(f"Gemma RoPE requires even head_dim, got {head_dim}")
            self._gemma_rope_cache[cache_key] = rope
        if int(batch_size) == 1:
            return rope
        return rope.repeat(int(batch_size), 1).contiguous()

    def _ensure_verify_fast_buffers(
        self,
        *,
        batch_k: int,
        prefix_len: int,
        suffix_len: int,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        ffn_hidden: int,
        action_dim: int,
        buffer_dtype: torch.dtype,
        device: torch.device,
    ) -> VerifyFastBuffers:
        keys_per_group = int(prefix_len) + int(suffix_len)
        total_queries = int(batch_k) * int(suffix_len) * int(num_heads)
        total_keys = int(batch_k) * keys_per_group
        key = (
            int(batch_k),
            int(prefix_len),
            int(suffix_len),
            int(hidden_size),
            int(head_dim),
            int(num_heads),
            int(ffn_hidden),
            int(action_dim),
            buffer_dtype,
        )
        if self._verify_fast_buffers is None or self._verify_fast_buffers_key != key:
            self._verify_fast_buffers = VerifyFastBuffers(
                decoder_x=torch.empty((int(batch_k) * int(suffix_len), int(hidden_size)), device=device, dtype=buffer_dtype),
                decoder_x_buf=torch.empty(
                    (int(batch_k) * max(0, int(suffix_len) - 1), int(hidden_size)),
                    device=device,
                    dtype=buffer_dtype,
                ),
                decoder_state_buf=torch.empty((int(batch_k), int(hidden_size)), device=device, dtype=buffer_dtype),
                decoder_norm_factor_buf=torch.empty((int(batch_k) * int(suffix_len),), device=device, dtype=buffer_dtype),
                decoder_q_buf=torch.empty((total_queries, int(head_dim)), device=device, dtype=buffer_dtype),
                decoder_attn_buf=torch.empty((total_queries, total_keys), device=device, dtype=buffer_dtype),
                decoder_hidden=torch.empty((int(batch_k) * int(suffix_len), int(ffn_hidden)), device=device, dtype=buffer_dtype),
                grouped_k_buf=torch.empty((total_keys, int(head_dim)), device=device, dtype=buffer_dtype),
                grouped_v_buf=torch.empty((total_keys, int(head_dim)), device=device, dtype=buffer_dtype),
                suffix_k_buf=torch.empty((int(batch_k) * int(suffix_len), int(head_dim)), device=device, dtype=buffer_dtype),
                suffix_v_buf=torch.empty((int(batch_k) * int(suffix_len), int(head_dim)), device=device, dtype=buffer_dtype),
                velocity_buf=torch.empty(
                    (int(batch_k) * max(0, int(suffix_len) - 1), int(action_dim)),
                    device=device,
                    dtype=buffer_dtype,
                ),
            )
            self._verify_fast_buffers_key = key
        return self._verify_fast_buffers

    def _ensure_verify_postprocess_buffers(
        self,
        *,
        batch_size: int,
        horizon: int,
        action_dim: int,
        device: torch.device,
    ) -> VerifyPostprocessBuffers:
        key = (int(batch_size), int(horizon), int(action_dim), str(device))
        if self._verify_postprocess_buffers is None or self._verify_postprocess_buffers_key != key:
            self._verify_postprocess_buffers = VerifyPostprocessBuffers(
                accepted_prefix_len=torch.empty((int(batch_size),), device=device, dtype=torch.int64),
                action_prefix_len=torch.empty((int(batch_size),), device=device, dtype=torch.int64),
                gripper_verify_stop_mask=torch.empty((int(batch_size),), device=device, dtype=torch.bool),
                gripper_switch_cut_mask=torch.empty((int(batch_size),), device=device, dtype=torch.bool),
                metrics=torch.empty((5,), device=device, dtype=torch.float32),
                actions=torch.empty((int(batch_size), int(horizon), int(action_dim)), device=device, dtype=torch.float32),
            )
            self._verify_postprocess_buffers_key = key
        return self._verify_postprocess_buffers

    def _timestep_tensor(self, *, t_list: Sequence[float], device: torch.device) -> torch.Tensor:
        values = tuple(float(t) for t in t_list)
        key = (values, str(device))
        timestep = self._verify_timestep_cache.get(key)
        if timestep is None:
            timestep = torch.tensor(values, device=device, dtype=torch.float32)
            self._verify_timestep_cache[key] = timestep
        return timestep

    def _attention_kernel(
        self,
        *,
        q_rows: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        suffix_k: torch.Tensor,
        suffix_v: torch.Tensor,
        key_valid: torch.Tensor,
        batch_size: int,
        query_len: int,
        prefix_len: int,
        suffix_len: int,
        num_heads: int,
        head_dim: int,
        block_suffix_for_query0: bool,
    ) -> torch.Tensor:
        total_rows = int(batch_size) * int(num_heads) * int(query_len)
        total_keys = int(prefix_len) + int(suffix_len)
        device = q_rows.device
        logits = torch.empty((total_rows, total_keys), device=device, dtype=torch.float32)
        probs = torch.empty_like(logits)
        out = torch.empty((total_rows, int(head_dim)), device=device, dtype=self.compute_dtype)
        key_valid = key_valid.to(device=device, dtype=torch.int8).contiguous()
        grid_logits = max(1, min(128, triton.cdiv(total_rows, 4) * triton.cdiv(total_keys, 32)))
        _attention_logits_kernel[(grid_logits,)](
            _to_kernel_tensor(q_rows, device=device),
            _to_kernel_tensor(prefix_k, device=device),
            _to_kernel_tensor(suffix_k, device=device),
            logits,
            total_rows=total_rows,
            total_keys=total_keys,
            prefix_len=int(prefix_len),
            suffix_len=int(suffix_len),
            head_dim=int(head_dim),
            num_heads=int(num_heads),
            query_len=int(query_len),
            BLOCK_ROWS=4,
            BLOCK_KEYS=32,
            BLOCK_K=_head_block_size(head_dim),
        )
        softmax_block = min(1024, _next_power_of_two(total_keys))
        _attention_softmax_mask_kernel[(max(1, triton.cdiv(total_rows, 4)),)](
            logits,
            key_valid,
            probs,
            total_rows=total_rows,
            total_keys=total_keys,
            num_heads=int(num_heads),
            query_len=int(query_len),
            prefix_len=int(prefix_len),
            block_suffix_for_query0=1 if block_suffix_for_query0 else 0,
            BLOCK_ROWS=4,
            BLOCK_SIZE=softmax_block,
        )
        grid_values = max(1, min(128, triton.cdiv(total_rows, 4) * triton.cdiv(head_dim, 32)))
        _attention_values_kernel[(grid_values,)](
            probs,
            _to_kernel_tensor(prefix_v, device=device),
            _to_kernel_tensor(suffix_v, device=device),
            out,
            total_rows=total_rows,
            total_keys=total_keys,
            prefix_len=int(prefix_len),
            suffix_len=int(suffix_len),
            head_dim=int(head_dim),
            num_heads=int(num_heads),
            query_len=int(query_len),
            BLOCK_ROWS=4,
            BLOCK_KEYS=32,
            BLOCK_COLS=_head_block_size(head_dim),
        )
        return out

    def _mix_with_timestep(self, *, noise: torch.Tensor, x0: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, dim = map(int, noise.shape)
        out = torch.empty((batch_size, horizon, dim), device=noise.device, dtype=self.compute_dtype)
        rows = batch_size * horizon
        grid = max(1, min(128, triton.cdiv(rows, 16) * triton.cdiv(dim, 32)))
        _mix_with_timestep_kernel[(grid,)](
            _to_kernel_tensor(noise.reshape(rows, dim), device=noise.device),
            _to_kernel_tensor(x0.reshape(rows, dim), device=noise.device),
            timestep.to(device=noise.device, dtype=torch.float32).contiguous(),
            out.view(rows, dim),
            rows=rows,
            cols=dim,
            horizon=horizon,
            BLOCK_ROWS=16,
            BLOCK_COLS=_head_block_size(dim if dim <= 32 else 32),
        )
        return out

    def _time_bias(self, *, time_biases: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        rows = int(timestep.shape[0])
        hidden = int(time_biases.shape[1])
        out = torch.empty((rows, hidden), device=timestep.device, dtype=self.compute_dtype)
        grid = max(1, min(128, triton.cdiv(rows, 8) * triton.cdiv(hidden, 32)))
        _interpolate_time_bias_kernel[(grid,)](
            _to_kernel_tensor(time_biases, device=timestep.device),
            timestep.to(device=timestep.device, dtype=torch.float32).contiguous(),
            out,
            rows=rows,
            steps=int(time_biases.shape[0]),
            hidden=hidden,
            BLOCK_ROWS=8,
            BLOCK_COLS=_default_block_size(hidden),
        )
        return out

    def _sinusoidal_time_embedding(self, *, timestep: torch.Tensor, hidden_size: int) -> torch.Tensor:
        device = timestep.device
        fraction = torch.linspace(0.0, 1.0, int(hidden_size) // 2, dtype=torch.float32, device=device)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        scaling = (1.0 / period) * (2.0 * torch.pi)
        sin_input = scaling[None, :] * timestep[:, None].to(dtype=torch.float32)
        time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
        return time_emb.to(dtype=torch.float32).contiguous()

    def _add_time_bias_silu(self, *, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        rows, hidden = map(int, x.shape)
        out = torch.empty((rows, hidden), device=x.device, dtype=self.compute_dtype)
        grid = max(1, min(128, triton.cdiv(rows, 8) * triton.cdiv(hidden, 32)))
        _add_rowwise_bias_silu_kernel[(grid,)](
            _to_kernel_tensor(x, device=x.device),
            _to_kernel_tensor(bias, device=x.device),
            out,
            rows=rows,
            hidden=hidden,
            BLOCK_ROWS=8,
            BLOCK_COLS=_default_block_size(hidden),
        )
        return out

    def _x0_hat_from_velocity(self, *, x_t: torch.Tensor, velocity: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, dim = map(int, x_t.shape)
        out = torch.empty((batch_size, horizon, dim), device=x_t.device, dtype=self.compute_dtype)
        rows = batch_size * horizon
        grid = max(1, min(128, triton.cdiv(rows, 16) * triton.cdiv(dim, 32)))
        _x0_hat_kernel[(grid,)](
            _to_kernel_tensor(x_t.reshape(rows, dim), device=x_t.device),
            _to_kernel_tensor(velocity.reshape(rows, dim), device=x_t.device),
            timestep.to(device=x_t.device, dtype=torch.float32).contiguous(),
            out.view(rows, dim),
            rows=rows,
            cols=dim,
            horizon=horizon,
            BLOCK_ROWS=16,
            BLOCK_COLS=_head_block_size(dim if dim <= 32 else 32),
        )
        return out

    def _prepare_prefix_from_full_runtime(
        self,
        *,
        images: torch.Tensor,
        state: torch.Tensor,
        full_runtime: Any,
    ) -> tuple[PrefixContext, float] | None:
        required = (
            "buffers",
            "_replay_or_run",
            "_encoder_graph",
            "_record_encoder_stage",
            "weights",
        )
        if full_runtime is None or not all(hasattr(full_runtime, name) for name in required):
            return None
        if images.ndim != 4:
            return None
        if state.ndim == 2 and int(state.shape[0]) != 1:
            return None

        runtime_device = full_runtime.buffers["observation_images_normalized"].device
        state_value = state[0] if state.ndim == 2 else state

        def _run_full_encoder() -> None:
            graph = getattr(full_runtime, "_encoder_graph", None)
            if graph is not None and hasattr(graph, "replay"):
                graph.replay()
            else:
                full_runtime._record_encoder_stage()

        def _run() -> PrefixContext:
            full_runtime.buffers["observation_images_normalized"].copy_(
                images.to(device=runtime_device, dtype=full_runtime.buffers["observation_images_normalized"].dtype)
            )
            full_runtime.buffers["observation_state_normalized"].copy_(
                state_value.to(device=runtime_device, dtype=full_runtime.buffers["observation_state_normalized"].dtype)
            )
            _run_full_encoder()
            vision_prefix = self._project_vision_prefix(full_runtime=full_runtime)
            language_embeds = self.weights["language_embeds"].to(device=vision_prefix.device, dtype=torch.float32)
            prefix_embs = torch.cat([vision_prefix, language_embeds.unsqueeze(0)], dim=1)
            seq_len = int(prefix_embs.shape[1])
            prefix_pad_masks = torch.ones((1, seq_len), device=prefix_embs.device, dtype=torch.bool)
            prefix_att_masks = torch.zeros((1, seq_len), device=prefix_embs.device, dtype=torch.bool)
            return PrefixContext(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
            )

        return _time_ms(_run, device=runtime_device)

    def _prepare_prefix_fallback(
        self,
        *,
        images: torch.Tensor,
        state: torch.Tensor,
    ) -> PrefixContext:
        del state
        if images.ndim == 4:
            batch_size = 1
            view_summary = images.to(device=self.device, dtype=torch.float32).mean(dim=(1, 2, 3), keepdim=False)
            vision_tokens = view_summary[:, None].expand(-1, self.hidden_size).unsqueeze(0)
        elif images.ndim == 5:
            batch_size = int(images.shape[0])
            view_summary = images.to(device=self.device, dtype=torch.float32).mean(dim=(2, 3, 4), keepdim=False)
            vision_tokens = view_summary.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        else:
            batch_size = 1
            vision_tokens = torch.zeros((1, self.num_views, self.hidden_size), device=self.device, dtype=torch.float32)

        language = self.weights["language_embeds"].to(dtype=vision_tokens.dtype)
        language = language.unsqueeze(0).expand(batch_size, -1, -1)
        prefix_embs = torch.cat([vision_tokens, language], dim=1).contiguous()
        prefix_pad_masks = torch.ones(prefix_embs.shape[:2], device=prefix_embs.device, dtype=torch.bool)
        prefix_att_masks = torch.zeros(prefix_embs.shape[:2], device=prefix_embs.device, dtype=torch.bool)
        return PrefixContext(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
        )

    def prepare_prefix(
        self,
        observation_images_normalized: torch.Tensor,
        observation_state_normalized: torch.Tensor,
        *,
        full_runtime: Any | None = None,
    ) -> PrefixContext:
        prepared = self._prepare_prefix_from_full_runtime(
            images=observation_images_normalized,
            state=observation_state_normalized,
            full_runtime=full_runtime,
        )
        if prepared is not None:
            return prepared[0]
        return self._prepare_prefix_fallback(
            images=observation_images_normalized,
            state=observation_state_normalized,
        )

    def _prepare_prefix_with_timing(
        self,
        observation_images_normalized: torch.Tensor,
        observation_state_normalized: torch.Tensor,
        *,
        full_runtime: Any | None = None,
    ) -> tuple[PrefixContext, float]:
        prepared = self._prepare_prefix_from_full_runtime(
            images=observation_images_normalized,
            state=observation_state_normalized,
            full_runtime=full_runtime,
        )
        if prepared is not None:
            prefix, encoder_ms = prepared
            return prefix, float(encoder_ms)

        prefix, encoder_ms = _time_ms(
            lambda: self._prepare_prefix_fallback(
                images=observation_images_normalized,
                state=observation_state_normalized,
            ),
            device=self.device,
        )
        return prefix, float(encoder_ms)

    def _draft_attention_mask(
        self,
        *,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        state_pad = torch.ones((int(prefix_pad_masks.shape[0]), 1), device=prefix_pad_masks.device, dtype=torch.bool)
        state_att = torch.zeros((int(prefix_pad_masks.shape[0]), 1), device=prefix_pad_masks.device, dtype=torch.bool)
        prefix_plus_state_pad = torch.cat([prefix_pad_masks.to(dtype=torch.bool), state_pad], dim=1)
        del state_att, prefix_att_masks
        total_prefix = int(prefix_plus_state_pad.shape[1])
        query_count = int(self.chunk_m)
        query_keys = torch.ones(
            (int(prefix_pad_masks.shape[0]), query_count),
            device=prefix_pad_masks.device,
            dtype=torch.bool,
        )
        return torch.cat([prefix_plus_state_pad, query_keys], dim=1).contiguous(), total_prefix

    def _run_draft_block(
        self,
        *,
        prefix: PrefixContext,
        observation_state_normalized: torch.Tensor,
    ) -> torch.Tensor:
        state = observation_state_normalized
        if state.ndim == 1:
            state = state.unsqueeze(0)
        device = prefix.prefix_embs.device
        state = state.to(device=device, dtype=torch.float32)
        prefix_embs = prefix.prefix_embs.to(device=device, dtype=torch.float32)
        device = prefix_embs.device
        b = int(prefix_embs.shape[0])
        state_token = _matmul_bias_kernel(
            state,
            self._draft["draft_state_in_proj_w"],
            self._draft["draft_state_in_proj_b"],
        ).view(b, 1, self.hidden_size)
        query_tokens = self._draft["draft_action_queries"][: self.chunk_m].to(device=device, dtype=torch.float32)
        query_tokens = query_tokens.unsqueeze(0).expand(b, -1, -1).contiguous()

        hidden_states = torch.cat(
            [
                prefix_embs.to(dtype=torch.float32),
                state_token.to(dtype=torch.float32),
                query_tokens,
            ],
            dim=1,
        )
        seq_len = int(hidden_states.shape[1])
        rope = self._gemma_rope_weights(seq_len=seq_len, batch_size=b, head_dim=self.head_dim, device=device)
        q_flat, k_flat, v_flat = _rms_qkv_rope_kernel(
            hidden_states.reshape(b * seq_len, self.hidden_size),
            self._draft["draft_qkv_w_scaled"],
            rope,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        q = q_flat.view(b, seq_len, self.num_heads, self.head_dim)
        k = k_flat.view(b, seq_len, self.head_dim)
        v = v_flat.view(b, seq_len, self.head_dim)
        key_valid, prefix_len = self._draft_attention_mask(
            prefix_pad_masks=prefix.prefix_pad_masks,
            prefix_att_masks=prefix.prefix_att_masks,
        )
        suffix_len = int(self.chunk_m)
        q_rows = (
            q[:, -suffix_len:, :, :]
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(b * self.num_heads * suffix_len, self.head_dim)
        )
        prefix_k = k[:, :prefix_len, :].contiguous().reshape(b * prefix_len, self.head_dim)
        prefix_v = v[:, :prefix_len, :].contiguous().reshape(b * prefix_len, self.head_dim)
        suffix_k = k[:, prefix_len:, :].contiguous().reshape(b * suffix_len, self.head_dim)
        suffix_v = v[:, prefix_len:, :].contiguous().reshape(b * suffix_len, self.head_dim)
        attn_rows = self._attention_kernel(
            q_rows=q_rows,
            prefix_k=prefix_k,
            prefix_v=prefix_v,
            suffix_k=suffix_k,
            suffix_v=suffix_v,
            key_valid=key_valid,
            batch_size=b,
            query_len=suffix_len,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_suffix_for_query0=False,
        )
        attn_out = (
            attn_rows.view(b, self.num_heads, suffix_len, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(b * suffix_len, self.num_heads * self.head_dim)
        )
        query_hidden = _matmul_residual_kernel(
            attn_out,
            self._draft["draft_attn_o_w"],
            query_tokens.reshape(b * suffix_len, self.hidden_size),
        )
        ff_hidden = _rms_matmul_gate_kernel(
            query_hidden,
            self._draft["draft_ffn_gate_w_scaled"],
            self._draft["draft_ffn_up_w_scaled"],
        )
        query_hidden = _matmul_residual_kernel(ff_hidden, self._draft["draft_ffn_down_w"], query_hidden)
        draft_actions = _matmul_bias_kernel(
            query_hidden,
            self._draft["draft_action_head_w"],
            self._draft["draft_action_head_b"],
        ).view(b, suffix_len, self.out_dim)

        x0_draft = torch.zeros(
            (b, self.chunk_size, 32),
            device=draft_actions.device,
            dtype=torch.float32,
        )
        x0_draft[:, : self.chunk_m, : self.out_dim] = draft_actions[:, : self.chunk_m, : self.out_dim]
        return x0_draft

    def _draft_graph_enabled(self, *, prefix: PrefixContext) -> bool:
        return (
            prefix.prefix_embs.device.type == "cuda"
            and torch.cuda.is_available()
            and _spec_triton_env("DRAFT_GRAPH", "1") != "0"
            and not self._draft_graph_failed
        )

    def _draft_graph_key(
        self,
        *,
        prefix: PrefixContext,
        observation_state_normalized: torch.Tensor,
    ) -> tuple[Any, ...]:
        return (
            str(prefix.prefix_embs.device),
            self.compute_dtype,
            tuple(prefix.prefix_embs.shape),
            tuple(prefix.prefix_pad_masks.shape),
            tuple(prefix.prefix_att_masks.shape),
            tuple(observation_state_normalized.shape),
            int(self.chunk_m),
            int(self.chunk_size),
            int(self.hidden_size),
            int(self.num_heads),
            int(self.head_dim),
        )

    @staticmethod
    def _copy_draft_graph_inputs(
        graph_state: _DraftGraph,
        *,
        prefix: PrefixContext,
        observation_state_normalized: torch.Tensor,
    ) -> None:
        graph_state.prefix_embs.copy_(
            prefix.prefix_embs.to(device=graph_state.prefix_embs.device, dtype=graph_state.prefix_embs.dtype)
        )
        graph_state.prefix_pad_masks.copy_(
            prefix.prefix_pad_masks.to(device=graph_state.prefix_pad_masks.device, dtype=graph_state.prefix_pad_masks.dtype)
        )
        graph_state.prefix_att_masks.copy_(
            prefix.prefix_att_masks.to(device=graph_state.prefix_att_masks.device, dtype=graph_state.prefix_att_masks.dtype)
        )
        graph_state.state.copy_(
            observation_state_normalized.to(device=graph_state.state.device, dtype=graph_state.state.dtype)
        )

    def _get_or_create_draft_graph(
        self,
        *,
        prefix: PrefixContext,
        observation_state_normalized: torch.Tensor,
    ) -> _DraftGraph:
        key = self._draft_graph_key(prefix=prefix, observation_state_normalized=observation_state_normalized)
        graph_state = self._draft_graph
        if graph_state is not None and graph_state.key == key:
            return graph_state

        device = prefix.prefix_embs.device
        static_prefix_embs = torch.empty_like(prefix.prefix_embs, device=device)
        static_prefix_pad_masks = torch.empty_like(prefix.prefix_pad_masks, device=device)
        static_prefix_att_masks = torch.empty_like(prefix.prefix_att_masks, device=device)
        static_state = torch.empty_like(
            observation_state_normalized.to(device=device, dtype=torch.float32),
            device=device,
            dtype=torch.float32,
        )
        static_prefix = PrefixContext(
            prefix_embs=static_prefix_embs,
            prefix_pad_masks=static_prefix_pad_masks,
            prefix_att_masks=static_prefix_att_masks,
        )
        graph_state = _DraftGraph(
            key=key,
            graph=torch.cuda.CUDAGraph(),
            prefix_embs=static_prefix_embs,
            prefix_pad_masks=static_prefix_pad_masks,
            prefix_att_masks=static_prefix_att_masks,
            state=static_state,
            prefix=static_prefix,
            output=torch.empty((1, self.chunk_size, 32), device=device, dtype=torch.float32),
        )
        self._copy_draft_graph_inputs(
            graph_state,
            prefix=prefix,
            observation_state_normalized=observation_state_normalized,
        )

        for _ in range(2):
            graph_state.output = self._run_draft_block(
                prefix=graph_state.prefix,
                observation_state_normalized=graph_state.state,
            )
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph_state.graph):
            graph_state.output = self._run_draft_block(
                prefix=graph_state.prefix,
                observation_state_normalized=graph_state.state,
            )
        torch.cuda.synchronize(device)
        self._draft_graph = graph_state
        return graph_state

    def _run_draft_block_maybe_graph(
        self,
        *,
        prefix: PrefixContext,
        observation_state_normalized: torch.Tensor,
    ) -> torch.Tensor:
        if self._draft_graph_enabled(prefix=prefix):
            try:
                graph_state = self._get_or_create_draft_graph(
                    prefix=prefix,
                    observation_state_normalized=observation_state_normalized,
                )
                self._copy_draft_graph_inputs(
                    graph_state,
                    prefix=prefix,
                    observation_state_normalized=observation_state_normalized,
                )
                graph_state.graph.replay()
                return graph_state.output
            except Exception:
                self._draft_graph = None
                self._draft_graph_failed = True
        return self._run_draft_block(
            prefix=prefix,
            observation_state_normalized=observation_state_normalized,
        )

    def run_draft(
        self,
        observation_images_normalized: torch.Tensor,
        observation_state_normalized: torch.Tensor,
        *,
        full_runtime: Any | None = None,
    ) -> torch.Tensor:
        prefix = self.prepare_prefix(
            observation_images_normalized,
            observation_state_normalized,
            full_runtime=full_runtime,
        )
        return self._run_draft_block_maybe_graph(
            prefix=prefix,
            observation_state_normalized=observation_state_normalized,
        )

    def run_draft_with_timing(
        self,
        observation_images_normalized: torch.Tensor,
        observation_state_normalized: torch.Tensor,
        *,
        full_runtime: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        prefix, encoder_ms = self._prepare_prefix_with_timing(
            observation_images_normalized,
            observation_state_normalized,
            full_runtime=full_runtime,
        )
        x0_draft, draft_ms = _time_ms(
            lambda: self._run_draft_block_maybe_graph(
                prefix=prefix,
                observation_state_normalized=observation_state_normalized,
            ),
            device=prefix.prefix_embs.device,
        )
        return x0_draft, {
            "encoder_ms": float(encoder_ms),
            "draft_ms": float(draft_ms),
        }

    def capture_full_cache_snapshot(self, full_runtime: Any) -> FullCacheSnapshot:
        if full_runtime is not None and hasattr(full_runtime, "buffers"):
            buffers = full_runtime.buffers
            if "encoder_K" in buffers and "encoder_V" in buffers and "encoder_x" in buffers:
                encoder_seq_len = int(getattr(full_runtime, "_encoder_seq_len", buffers["encoder_x"].shape[0]))
                return FullCacheSnapshot(
                    encoder_seq_len=encoder_seq_len,
                    encoder_x=buffers["encoder_x"][:encoder_seq_len].detach().clone(),
                    encoder_k=buffers["encoder_K"][:, :encoder_seq_len].detach().clone(),
                    encoder_v=buffers["encoder_V"][:, :encoder_seq_len].detach().clone(),
                )
        return FullCacheSnapshot(encoder_seq_len=int(self.num_views + self.prompt_len))

    def _build_verify_batch(
        self,
        *,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: Sequence[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise.ndim == 2:
            noise = noise.unsqueeze(0)
        if x0_draft.ndim == 2:
            x0_draft = x0_draft.unsqueeze(0)
        if tuple(noise.shape) != tuple(x0_draft.shape):
            raise ValueError(f"noise shape={tuple(noise.shape)} must match x0_draft shape={tuple(x0_draft.shape)}")

        timestep = self._timestep_tensor(t_list=t_list, device=x0_draft.device)
        k = int(timestep.shape[0])
        noise_bk = _expand_batch(noise.to(dtype=torch.float32), k)
        x0_bk = _expand_batch(x0_draft.to(dtype=torch.float32), k)
        timestep_bk = timestep.unsqueeze(0).expand(int(noise.shape[0]), -1).reshape(-1)
        x_t_bk = self._mix_with_timestep(
            noise=noise_bk.to(device=self.device, dtype=torch.float32),
            x0=x0_bk.to(device=self.device, dtype=torch.float32),
            timestep=timestep_bk.to(device=self.device, dtype=torch.float32),
        )
        return x_t_bk.to(dtype=torch.float32).contiguous(), timestep_bk.contiguous()

    @staticmethod
    def _decoder_required_keys() -> tuple[str, ...]:
        return (
            "decoder_state_in_proj_w",
            "decoder_state_in_proj_b",
            "decoder_action_in_proj_w",
            "decoder_action_in_proj_b",
            "decoder_action_time_mlp_in_w",
            "decoder_action_time_mlp_in_b",
            "decoder_action_fused_in_proj_w",
            "decoder_action_fused_time_biases",
            "decoder_action_mlp_w",
            "decoder_action_mlp_b",
            "decoder_attn_qkv_w",
            "decoder_attn_o_w",
            "decoder_ffn_gate_w",
            "decoder_ffn_up_w",
            "decoder_ffn_down_w",
            "decoder_action_fused_out_proj_w",
            "decoder_action_fused_out_proj_b",
        )

    def _full_verify_unavailable_reasons(
        self,
        *,
        full_runtime: Any | None,
        cache_snapshot: FullCacheSnapshot,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if full_runtime is None:
            reasons.append("full_runtime is None")
            return tuple(reasons)

        weights = getattr(full_runtime, "weights", None)
        if not isinstance(weights, Mapping):
            reasons.append("full_runtime.weights is unavailable")
        else:
            missing = [key for key in self._decoder_required_keys() if key not in weights]
            if missing:
                reasons.append(f"missing decoder weights: {', '.join(missing)}")
        if cache_snapshot.encoder_k is None:
            reasons.append("cache_snapshot.encoder_k is missing")
        if cache_snapshot.encoder_v is None:
            reasons.append("cache_snapshot.encoder_v is missing")
        return tuple(reasons)

    @staticmethod
    def _triton_delta_to_velocity_scale(*, weights: Mapping[str, torch.Tensor]) -> float:
        time_biases = weights.get("decoder_action_fused_time_biases")
        if isinstance(time_biases, torch.Tensor) and int(time_biases.shape[0]) > 0:
            return -float(time_biases.shape[0])
        return -10.0

    @staticmethod
    def _verify_exact_input_supported(*, weights: Mapping[str, torch.Tensor]) -> bool:
        required = (
            "decoder_action_in_proj_w",
            "decoder_action_in_proj_b",
            "decoder_action_time_mlp_in_w",
            "decoder_action_time_mlp_in_b",
            "decoder_action_mlp_w",
            "decoder_action_mlp_b",
        )
        return all(key in weights for key in required)

    @staticmethod
    def _verify_exact_fp32_supported(
        *,
        weights: Mapping[str, torch.Tensor],
        x_t_bk: torch.Tensor,
        hidden_size: int,
        action_dim: int,
    ) -> bool:
        if _spec_triton_env("VERIFY_EXACT_FP32", "0") != "1":
            return False
        if x_t_bk.device.type != "cuda":
            return False
        if int(hidden_size) != 1024 or int(action_dim) != 32:
            return False
        if not Pi0SpecInference._verify_exact_input_supported(weights=weights):
            return False
        signal_keys = (
            "decoder_action_time_mlp_in_w",
            "decoder_action_mlp_w",
            "decoder_action_fused_out_proj_w",
        )
        return any(bool(weights[key].abs().any().item()) for key in signal_keys)

    def _build_verify_action_hidden_exact(
        self,
        *,
        weights: Mapping[str, torch.Tensor],
        x_t_bk: torch.Tensor,
        timestep_bk: torch.Tensor,
        action_horizon: int,
        action_dim: int,
        hidden_size: int,
        device: torch.device,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = int(x_t_bk.shape[0]) * int(action_horizon)
        action_emb = _matmul_bias_kernel(
            x_t_bk.reshape(rows, action_dim),
            weights["decoder_action_in_proj_w"],
            weights["decoder_action_in_proj_b"],
        )
        time_emb = self._sinusoidal_time_embedding(
            timestep=timestep_bk.to(device=device, dtype=torch.float32),
            hidden_size=hidden_size,
        )
        mlp_in_w = weights["decoder_action_time_mlp_in_w"]
        action_w = mlp_in_w[:hidden_size]
        time_w = mlp_in_w[hidden_size:]
        preact = _matmul_kernel(action_emb, action_w)
        time_term = _matmul_kernel(_expand_batch(time_emb, action_horizon).contiguous(), time_w)
        bias = time_term + weights["decoder_action_time_mlp_in_b"].to(device=device, dtype=torch.float32)
        hidden = self._add_time_bias_silu(x=preact, bias=bias)
        return _matmul_bias_kernel(
            hidden,
            weights["decoder_action_mlp_w"],
            weights["decoder_action_mlp_b"],
            out=out,
        )

    def _build_verify_action_hidden_exact_fp32(
        self,
        *,
        weights: Mapping[str, torch.Tensor],
        x_t_bk: torch.Tensor,
        timestep_bk: torch.Tensor,
        action_horizon: int,
        action_dim: int,
        hidden_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        rows = int(x_t_bk.shape[0]) * int(action_horizon)
        action_emb = _matmul_bias_fp32(
            x_t_bk.reshape(rows, action_dim),
            weights["decoder_action_in_proj_w"],
            weights["decoder_action_in_proj_b"],
        )
        time_emb = self._sinusoidal_time_embedding(
            timestep=timestep_bk.to(device=device, dtype=torch.float32),
            hidden_size=hidden_size,
        )
        mlp_in_w = weights["decoder_action_time_mlp_in_w"]
        action_w = mlp_in_w[:hidden_size]
        time_w = mlp_in_w[hidden_size:]
        preact = _matmul_fp32(action_emb, action_w)
        time_term = _matmul_fp32(_expand_batch(time_emb, action_horizon).contiguous(), time_w)
        bias = time_term + weights["decoder_action_time_mlp_in_b"].to(device=device, dtype=torch.float32)
        hidden = torch.nn.functional.silu(preact + bias)
        return _matmul_bias_fp32(
            hidden,
            weights["decoder_action_mlp_w"],
            weights["decoder_action_mlp_b"],
        )

    def _build_verify_action_hidden_fused(
        self,
        *,
        weights: Mapping[str, torch.Tensor],
        x_t_bk: torch.Tensor,
        timestep_bk: torch.Tensor,
        action_horizon: int,
        action_dim: int,
        device: torch.device,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows = int(x_t_bk.shape[0]) * int(action_horizon)
        action_hidden = _matmul_kernel(
            x_t_bk.reshape(rows, action_dim),
            weights["decoder_action_fused_in_proj_w"],
            out=out,
        )
        time_bias = self._time_bias(
            time_biases=weights["decoder_action_fused_time_biases"],
            timestep=timestep_bk.to(device=device, dtype=torch.float32),
        )
        time_bias = _expand_batch(time_bias, action_horizon).contiguous()
        action_hidden = self._add_time_bias_silu(x=action_hidden, bias=time_bias)
        return _matmul_bias_kernel(
            action_hidden,
            weights["decoder_action_mlp_w"],
            weights["decoder_action_mlp_b"],
            out=out,
        )

    def _build_verify_action_hidden(
        self,
        *,
        weights: Mapping[str, torch.Tensor],
        x_t_bk: torch.Tensor,
        timestep_bk: torch.Tensor,
        action_horizon: int,
        action_dim: int,
        hidden_size: int,
        device: torch.device,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            _spec_triton_env("VERIFY_EXACT_INPUT", "0") == "1"
            and self._verify_exact_input_supported(weights=weights)
        ):
            return self._build_verify_action_hidden_exact(
                weights=weights,
                x_t_bk=x_t_bk,
                timestep_bk=timestep_bk,
                action_horizon=action_horizon,
                action_dim=action_dim,
                hidden_size=hidden_size,
                device=device,
                out=out,
            )
        return self._build_verify_action_hidden_fused(
            weights=weights,
            x_t_bk=x_t_bk,
            timestep_bk=timestep_bk,
            action_horizon=action_horizon,
            action_dim=action_dim,
            device=device,
            out=out,
        )

    @staticmethod
    def _verify_fast_path_supported(
        *,
        weights: Mapping[str, torch.Tensor],
        cache_snapshot: FullCacheSnapshot,
        x_t_bk: torch.Tensor,
    ) -> bool:
        if x_t_bk.device.type != "cuda" or cache_snapshot.encoder_k is None or cache_snapshot.encoder_v is None:
            return False
        exact_required = {
            "decoder_action_in_proj_w": (32, 1024),
            "decoder_action_in_proj_b": (1024,),
            "decoder_action_time_mlp_in_w": (2048, 1024),
            "decoder_action_time_mlp_in_b": (1024,),
        }
        for key, expected in exact_required.items():
            if key not in weights:
                return False
            if tuple(weights[key].shape[-len(expected) :]) != expected:
                return False
        expected_shapes = {
            "decoder_state_in_proj_w": (32, 1024),
            "decoder_state_in_proj_b": (1024,),
            "decoder_action_fused_in_proj_w": (32, 1024),
            "decoder_action_fused_time_biases": (10, 1024),
            "decoder_action_mlp_w": (1024, 1024),
            "decoder_action_mlp_b": (1024,),
            "decoder_attn_qkv_w": (1024, 2560),
            "decoder_attn_o_w": (2048, 1024),
            "decoder_ffn_gate_w": (1024, 4096),
            "decoder_ffn_up_w": (1024, 4096),
            "decoder_ffn_down_w": (4096, 1024),
            "decoder_action_fused_out_proj_w": (1024, 32),
            "decoder_action_fused_out_proj_b": (32,),
        }
        for key, expected in expected_shapes.items():
            if key not in weights:
                return False
            shape = tuple(weights[key].shape[-len(expected) :])
            if shape != expected:
                return False
        if tuple(cache_snapshot.encoder_k.shape[1:]) != (int(cache_snapshot.encoder_seq_len), 256):
            return False
        if tuple(cache_snapshot.encoder_v.shape[1:]) != (int(cache_snapshot.encoder_seq_len), 256):
            return False
        suffix_len = int(x_t_bk.shape[1]) + 1
        batch_k = int(x_t_bk.shape[0])
        total_keys = batch_k * (int(cache_snapshot.encoder_seq_len) + suffix_len)
        return total_keys <= 4096

    def _pack_grouped_kv(
        self,
        *,
        prefix_kv: torch.Tensor,
        suffix_kv: torch.Tensor,
        out: torch.Tensor,
        batch_k: int,
        prefix_len: int,
        suffix_len: int,
        head_dim: int,
    ) -> torch.Tensor:
        grid = max(1, min(128, triton.cdiv(int(batch_k) * (int(prefix_len) + int(suffix_len)), 8) * triton.cdiv(int(head_dim), 32)))
        _pack_grouped_kv_kernel[(grid,)](
            _to_kernel_tensor(prefix_kv, device=out.device),
            _to_kernel_tensor(suffix_kv, device=out.device),
            out,
            num_groups=int(batch_k),
            prefix_len=int(prefix_len),
            suffix_len=int(suffix_len),
            head_dim=int(head_dim),
            BLOCK_ROWS=8,
            BLOCK_COLS=_head_block_size(head_dim),
        )
        return out

    @staticmethod
    def _rms_matmul_gate_verify_spec(
        x: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        out: torch.Tensor,
        norm_factor: torch.Tensor,
    ) -> None:
        seq_len = int(x.shape[0])
        if seq_len != 102 or x.dtype != torch.bfloat16:
            _PI0_INFER.rms_matmul_k_1024_4096_gate(x, weight1, weight2, out, norm_factor)
            return
        _PI0_INFER.rmsnorm_factor_kernel[(128,)](x, norm_factor, seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024)
        _PI0_INFER.scaled_matmul_small_gate[(128,)](
            x,
            norm_factor,
            weight1,
            weight2,
            out,
            seq_len=seq_len,
            features=1024,
            hidden=4096,
            BLOCK_SIZE_N=16,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_K=64,
        )

    @staticmethod
    def _matmul_down_verify_spec(
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        seq_len = int(x.shape[0])
        if seq_len != 102 or x.dtype != torch.bfloat16:
            _PI0_INFER.matmul_k_4096_1024_res(x, weight, out)
            return
        _PI0_INFER.matmul_small_res[(((seq_len + 15) // 16) * (1024 // 64),)](
            x,
            weight,
            out,
            out,
            seq_len=seq_len,
            features=4096,
            hidden=1024,
            BLOCK_SIZE_N=16,
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_K=256,
        )

    def _run_batched_verify_fast(
        self,
        *,
        full_runtime: Any,
        cache_snapshot: FullCacheSnapshot,
        observation_state_normalized: torch.Tensor,
        x_t_bk: torch.Tensor,
        timestep_bk: torch.Tensor,
    ) -> torch.Tensor:
        weights = getattr(full_runtime, "weights", None)
        if not isinstance(weights, Mapping):
            raise ValueError("full runtime weights are unavailable for verify")
        if cache_snapshot.encoder_k is None or cache_snapshot.encoder_v is None:
            raise ValueError("full cache snapshot is missing encoder KV tensors")

        device = x_t_bk.device
        bk = int(x_t_bk.shape[0])
        action_horizon = int(x_t_bk.shape[1])
        action_dim = int(x_t_bk.shape[2])
        suffix_len = int(action_horizon + 1)
        prefix_len = int(cache_snapshot.encoder_seq_len)
        num_heads = 8
        head_dim = 256
        hidden_size = 1024
        ffn_hidden = 4096
        q_dim = int(num_heads * head_dim)
        decoder_layers = int(weights["decoder_attn_qkv_w"].shape[0])
        keys_per_group = int(prefix_len + suffix_len)
        total_queries = int(bk * suffix_len * num_heads)
        total_keys = int(bk * keys_per_group)

        use_fp32_exact = self._verify_exact_fp32_supported(
            weights=weights,
            x_t_bk=x_t_bk,
            hidden_size=hidden_size,
            action_dim=action_dim,
        )
        buffer_dtype = torch.float32 if use_fp32_exact else self.compute_dtype
        buffers = self._ensure_verify_fast_buffers(
            batch_k=bk,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            ffn_hidden=ffn_hidden,
            action_dim=action_dim,
            buffer_dtype=buffer_dtype,
            device=device,
        )

        state = observation_state_normalized
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state_bk = _expand_batch(
            state.to(device=device, dtype=torch.float32),
            int(timestep_bk.shape[0]) // int(state.shape[0]),
        )

        if use_fp32_exact:
            state_hidden = _matmul_bias_fp32(
                state_bk,
                weights["decoder_state_in_proj_w"],
                weights["decoder_state_in_proj_b"],
            )
            action_hidden = self._build_verify_action_hidden_exact_fp32(
                weights=weights,
                x_t_bk=x_t_bk,
                timestep_bk=timestep_bk,
                action_horizon=action_horizon,
                action_dim=action_dim,
                hidden_size=hidden_size,
                device=device,
            )
        else:
            state_hidden = _matmul_bias_kernel(
                state_bk,
                weights["decoder_state_in_proj_w"],
                weights["decoder_state_in_proj_b"],
                out=buffers.decoder_state_buf,
            )
            action_hidden = self._build_verify_action_hidden(
                weights=weights,
                x_t_bk=x_t_bk,
                timestep_bk=timestep_bk,
                action_horizon=action_horizon,
                action_dim=action_dim,
                hidden_size=hidden_size,
                device=device,
                out=buffers.decoder_x_buf,
            )

        decoder_x = buffers.decoder_x
        decoder_x_view = decoder_x.view(bk, suffix_len, hidden_size)
        decoder_x_view[:, 0, :].copy_(state_hidden.view(bk, hidden_size))
        decoder_x_view[:, 1:, :].copy_(action_hidden.view(bk, action_horizon, hidden_size))

        encoder_k = _to_kernel_tensor(cache_snapshot.encoder_k, device=device, dtype=buffer_dtype)
        encoder_v = _to_kernel_tensor(cache_snapshot.encoder_v, device=device, dtype=buffer_dtype)
        attn_qkv_w = _to_kernel_tensor(weights["decoder_attn_qkv_w"], device=device, dtype=buffer_dtype)
        attn_o_w = _to_kernel_tensor(weights["decoder_attn_o_w"], device=device, dtype=buffer_dtype)
        ffn_gate_w = _to_kernel_tensor(weights["decoder_ffn_gate_w"], device=device, dtype=buffer_dtype)
        ffn_up_w = _to_kernel_tensor(weights["decoder_ffn_up_w"], device=device, dtype=buffer_dtype)
        ffn_down_w = _to_kernel_tensor(weights["decoder_ffn_down_w"], device=device, dtype=buffer_dtype)
        out_proj_w = _to_kernel_tensor(weights["decoder_action_fused_out_proj_w"], device=device, dtype=buffer_dtype)
        out_proj_b = _to_kernel_tensor(weights["decoder_action_fused_out_proj_b"], device=device, dtype=buffer_dtype)
        rope = self._rope_weights(
            seq_len=suffix_len,
            batch_size=bk,
            head_dim=head_dim,
            device=device,
            position_offset=prefix_len,
        )
        if use_fp32_exact:
            rope = rope.to(device=device, dtype=torch.float32).contiguous()
        softmax_block = _next_power_of_two(total_keys)

        for layer_idx in range(decoder_layers):
            _PI0_INFER.rms_matmul_k_1024_2560_qkv_rope(
                decoder_x,
                attn_qkv_w[layer_idx],
                rope,
                buffers.decoder_q_buf,
                buffers.suffix_k_buf,
                buffers.suffix_v_buf,
                buffers.decoder_norm_factor_buf,
            )
            self._pack_grouped_kv(
                prefix_kv=encoder_k[layer_idx],
                suffix_kv=buffers.suffix_k_buf,
                out=buffers.grouped_k_buf,
                batch_k=bk,
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                head_dim=head_dim,
            )
            self._pack_grouped_kv(
                prefix_kv=encoder_v[layer_idx],
                suffix_kv=buffers.suffix_v_buf,
                out=buffers.grouped_v_buf,
                batch_k=bk,
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                head_dim=head_dim,
            )
            _PI0_INFER.matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
                buffers.decoder_q_buf,
                buffers.grouped_k_buf,
                buffers.decoder_attn_buf,
                total_queries,
                total_keys,
                head_dim,
                head_dim ** -0.5,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=64,
            )
            _grouped_softmax_mask0_kernel[(max(1, triton.cdiv(total_queries, 4)),)](
                buffers.decoder_attn_buf,
                buffers.decoder_attn_buf,
                total_queries=total_queries,
                keys_per_group=keys_per_group,
                num_groups=bk,
                num_heads=num_heads,
                query_len=suffix_len,
                prefix_len=prefix_len,
                block_suffix_for_query0=1,
                BLOCK_ROWS=4,
                BLOCK_SIZE=softmax_block,
            )
            _PI0_INFER.matmul_k8_n_256(
                buffers.decoder_attn_buf,
                buffers.grouped_v_buf,
                buffers.decoder_q_buf,
            )
            _PI0_INFER.matmul_k_2048_1024_res(
                buffers.decoder_q_buf.view(-1, q_dim),
                attn_o_w[layer_idx],
                decoder_x,
            )
            self._rms_matmul_gate_verify_spec(
                decoder_x,
                ffn_gate_w[layer_idx],
                ffn_up_w[layer_idx],
                buffers.decoder_hidden,
                buffers.decoder_norm_factor_buf,
            )
            self._matmul_down_verify_spec(
                buffers.decoder_hidden,
                ffn_down_w[layer_idx],
                decoder_x,
            )

        if use_fp32_exact:
            delta = _matmul_bias_fp32(
                _rms_norm_fp32(decoder_x_view[:, 1:, :].reshape(bk * action_horizon, hidden_size)),
                weights["decoder_action_fused_out_proj_w"],
                weights["decoder_action_fused_out_proj_b"],
            ).view(bk, action_horizon, action_dim)
        else:
            buffers.velocity_buf.zero_()
            _PI0_INFER.rms_matmul_k_1024_32_bias_res(
                decoder_x_view[:, 1:, :].reshape(bk * action_horizon, hidden_size),
                out_proj_w,
                out_proj_b,
                buffers.velocity_buf,
                buffers.decoder_norm_factor_buf[: bk * action_horizon],
            )
            delta = buffers.velocity_buf.view(bk, action_horizon, action_dim)
        velocity_scale = self._triton_delta_to_velocity_scale(weights=weights)
        velocity = delta.to(device=device, dtype=torch.float32) * float(velocity_scale)
        if use_fp32_exact:
            x0_hat = x_t_bk.to(device=device, dtype=torch.float32) - timestep_bk.to(device=device, dtype=torch.float32)[
                :, None, None
            ] * velocity
        else:
            x0_hat = self._x0_hat_from_velocity(
                x_t=x_t_bk.to(device=device, dtype=torch.float32),
                velocity=velocity,
                timestep=timestep_bk.to(device=device, dtype=torch.float32),
            )
        return x0_hat.to(dtype=torch.float32)

    def _verify_fast_graph_enabled(self) -> bool:
        return (
            self.device.type == "cuda"
            and torch.cuda.is_available()
            and _spec_triton_env("VERIFY_GRAPH", "1") != "0"
            and _spec_triton_env("VERIFY_EXACT_INPUT", "0") != "1"
            and _spec_triton_env("VERIFY_EXACT_FP32", "0") != "1"
            and not self._verify_fast_graph_failed
        )

    def _verify_fast_graph_key(
        self,
        *,
        weights: Mapping[str, torch.Tensor],
        cache_snapshot: FullCacheSnapshot,
        observation_state_normalized: torch.Tensor,
        x_t_bk: torch.Tensor,
        timestep_bk: torch.Tensor,
    ) -> tuple[Any, ...]:
        if cache_snapshot.encoder_k is None or cache_snapshot.encoder_v is None:
            raise ValueError("full cache snapshot is missing encoder KV tensors")
        return (
            id(weights),
            str(x_t_bk.device),
            self.compute_dtype,
            int(cache_snapshot.encoder_seq_len),
            tuple(cache_snapshot.encoder_k.shape),
            tuple(cache_snapshot.encoder_v.shape),
            tuple(observation_state_normalized.shape),
            tuple(x_t_bk.shape),
            tuple(timestep_bk.shape),
            tuple(weights["decoder_attn_qkv_w"].shape),
        )

    @staticmethod
    def _copy_verify_fast_graph_inputs(
        graph_state: _VerifyFastGraph,
        *,
        cache_snapshot: FullCacheSnapshot,
        observation_state_normalized: torch.Tensor,
        x_t_bk: torch.Tensor,
        timestep_bk: torch.Tensor,
    ) -> None:
        if cache_snapshot.encoder_k is None or cache_snapshot.encoder_v is None:
            raise ValueError("full cache snapshot is missing encoder KV tensors")
        graph_state.x_t_bk.copy_(x_t_bk.to(device=graph_state.x_t_bk.device, dtype=graph_state.x_t_bk.dtype))
        graph_state.timestep_bk.copy_(
            timestep_bk.to(device=graph_state.timestep_bk.device, dtype=graph_state.timestep_bk.dtype)
        )
        graph_state.state.copy_(
            observation_state_normalized.to(device=graph_state.state.device, dtype=graph_state.state.dtype)
        )
        encoder_k_source = (id(cache_snapshot.encoder_k), int(cache_snapshot.encoder_k.data_ptr()))
        encoder_v_source = (id(cache_snapshot.encoder_v), int(cache_snapshot.encoder_v.data_ptr()))
        if graph_state.encoder_k_source != encoder_k_source:
            graph_state.encoder_k.copy_(
                cache_snapshot.encoder_k.to(device=graph_state.encoder_k.device, dtype=graph_state.encoder_k.dtype)
            )
            graph_state.encoder_k_source = encoder_k_source
        if graph_state.encoder_v_source != encoder_v_source:
            graph_state.encoder_v.copy_(
                cache_snapshot.encoder_v.to(device=graph_state.encoder_v.device, dtype=graph_state.encoder_v.dtype)
            )
            graph_state.encoder_v_source = encoder_v_source

    def _get_or_create_verify_fast_graph(
        self,
        *,
        full_runtime: Any,
        cache_snapshot: FullCacheSnapshot,
        observation_state_normalized: torch.Tensor,
        x_t_bk: torch.Tensor,
        timestep_bk: torch.Tensor,
    ) -> _VerifyFastGraph:
        weights = getattr(full_runtime, "weights", None)
        if not isinstance(weights, Mapping):
            raise ValueError("full runtime weights are unavailable for verify")
        if cache_snapshot.encoder_k is None or cache_snapshot.encoder_v is None:
            raise ValueError("full cache snapshot is missing encoder KV tensors")

        key = self._verify_fast_graph_key(
            weights=weights,
            cache_snapshot=cache_snapshot,
            observation_state_normalized=observation_state_normalized,
            x_t_bk=x_t_bk,
            timestep_bk=timestep_bk,
        )
        graph_state = self._verify_fast_graph
        if graph_state is not None and graph_state.key == key:
            return graph_state

        device = x_t_bk.device
        static_x_t = torch.empty_like(x_t_bk, device=device, dtype=torch.float32)
        static_timestep = torch.empty_like(timestep_bk, device=device, dtype=torch.float32)
        static_state = torch.empty_like(
            observation_state_normalized.to(device=device, dtype=torch.float32),
            device=device,
            dtype=torch.float32,
        )
        static_encoder_k = torch.empty_like(cache_snapshot.encoder_k, device=device, dtype=self.compute_dtype)
        static_encoder_v = torch.empty_like(cache_snapshot.encoder_v, device=device, dtype=self.compute_dtype)
        static_cache = FullCacheSnapshot(
            encoder_seq_len=int(cache_snapshot.encoder_seq_len),
            encoder_x=None,
            encoder_k=static_encoder_k,
            encoder_v=static_encoder_v,
        )
        static_runtime = _VerifyGraphRuntime(weights=weights)
        graph_state = _VerifyFastGraph(
            key=key,
            graph=torch.cuda.CUDAGraph(),
            x_t_bk=static_x_t,
            timestep_bk=static_timestep,
            state=static_state,
            encoder_k=static_encoder_k,
            encoder_v=static_encoder_v,
            cache_snapshot=static_cache,
            full_runtime=static_runtime,
            buffers=self._ensure_verify_fast_buffers(
                batch_k=int(x_t_bk.shape[0]),
                prefix_len=int(cache_snapshot.encoder_seq_len),
                suffix_len=int(x_t_bk.shape[1]) + 1,
                hidden_size=1024,
                head_dim=256,
                num_heads=8,
                ffn_hidden=4096,
                action_dim=int(x_t_bk.shape[2]),
                buffer_dtype=self.compute_dtype,
                device=device,
            ),
            output=static_x_t,
        )
        self._copy_verify_fast_graph_inputs(
            graph_state,
            cache_snapshot=cache_snapshot,
            observation_state_normalized=observation_state_normalized,
            x_t_bk=x_t_bk,
            timestep_bk=timestep_bk,
        )

        for _ in range(2):
            graph_state.output = self._run_batched_verify_fast(
                full_runtime=graph_state.full_runtime,
                cache_snapshot=graph_state.cache_snapshot,
                observation_state_normalized=graph_state.state,
                x_t_bk=graph_state.x_t_bk,
                timestep_bk=graph_state.timestep_bk,
            )
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph_state.graph):
            graph_state.output = self._run_batched_verify_fast(
                full_runtime=graph_state.full_runtime,
                cache_snapshot=graph_state.cache_snapshot,
                observation_state_normalized=graph_state.state,
                x_t_bk=graph_state.x_t_bk,
                timestep_bk=graph_state.timestep_bk,
            )
        torch.cuda.synchronize(device)
        self._verify_fast_graph = graph_state
        return graph_state

    def _run_batched_verify_fast_graph(
        self,
        *,
        full_runtime: Any,
        cache_snapshot: FullCacheSnapshot,
        observation_state_normalized: torch.Tensor,
        x_t_bk: torch.Tensor,
        timestep_bk: torch.Tensor,
    ) -> torch.Tensor:
        graph_state = self._get_or_create_verify_fast_graph(
            full_runtime=full_runtime,
            cache_snapshot=cache_snapshot,
            observation_state_normalized=observation_state_normalized,
            x_t_bk=x_t_bk,
            timestep_bk=timestep_bk,
        )
        self._copy_verify_fast_graph_inputs(
            graph_state,
            cache_snapshot=cache_snapshot,
            observation_state_normalized=observation_state_normalized,
            x_t_bk=x_t_bk,
            timestep_bk=timestep_bk,
        )
        graph_state.graph.replay()
        return graph_state.output

    def _run_batched_verify_generic(
        self,
        *,
        full_runtime: Any,
        cache_snapshot: FullCacheSnapshot,
        observation_state_normalized: torch.Tensor,
        x_t_bk: torch.Tensor,
        timestep_bk: torch.Tensor,
    ) -> torch.Tensor:
        weights = getattr(full_runtime, "weights", None)
        if not isinstance(weights, Mapping):
            raise ValueError("full runtime weights are unavailable for verify")
        if cache_snapshot.encoder_k is None or cache_snapshot.encoder_v is None:
            raise ValueError("full cache snapshot is missing encoder KV tensors")

        device = x_t_bk.device
        bk = int(x_t_bk.shape[0])
        action_horizon = int(x_t_bk.shape[1])
        action_dim = int(x_t_bk.shape[2])
        state = observation_state_normalized
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state_bk = _expand_batch(
            state.to(device=device, dtype=torch.float32),
            int(timestep_bk.shape[0]) // int(state.shape[0]),
        )
        hidden_size = int(weights["decoder_state_in_proj_w"].shape[1])
        use_fp32_exact = self._verify_exact_fp32_supported(
            weights=weights,
            x_t_bk=x_t_bk,
            hidden_size=hidden_size,
            action_dim=action_dim,
        )
        if use_fp32_exact:
            state_hidden = _matmul_bias_fp32(
                state_bk,
                weights["decoder_state_in_proj_w"],
                weights["decoder_state_in_proj_b"],
            ).view(bk, hidden_size)
        else:
            state_hidden = _matmul_bias_kernel(
                state_bk,
                weights["decoder_state_in_proj_w"],
                weights["decoder_state_in_proj_b"],
            ).view(bk, hidden_size)
        if use_fp32_exact:
            action_hidden = self._build_verify_action_hidden_exact_fp32(
                weights=weights,
                x_t_bk=x_t_bk,
                timestep_bk=timestep_bk,
                action_horizon=action_horizon,
                action_dim=action_dim,
                hidden_size=hidden_size,
                device=device,
            ).view(bk, action_horizon, hidden_size)
        else:
            action_hidden = self._build_verify_action_hidden(
                weights=weights,
                x_t_bk=x_t_bk,
                timestep_bk=timestep_bk,
                action_horizon=action_horizon,
                action_dim=action_dim,
                hidden_size=hidden_size,
                device=device,
            ).view(bk, action_horizon, hidden_size)

        suffix_len = int(action_horizon + 1)
        x_dtype = torch.float32 if use_fp32_exact else self.compute_dtype
        x = torch.empty((bk, suffix_len, hidden_size), device=device, dtype=x_dtype)
        x[:, 0, :].copy_(state_hidden)
        x[:, 1:, :].copy_(action_hidden)
        prefix_len = int(cache_snapshot.encoder_seq_len)
        decoder_layers = int(weights["decoder_attn_qkv_w"].shape[0])
        encoder_dtype = torch.float32 if use_fp32_exact else self.compute_dtype
        encoder_k = cache_snapshot.encoder_k.to(device=device, dtype=encoder_dtype)
        encoder_v = cache_snapshot.encoder_v.to(device=device, dtype=encoder_dtype)
        kv_head_dim = int(encoder_k.shape[-1])
        qkv_out_dim = int(weights["decoder_attn_qkv_w"].shape[-1])
        q_dim = int(qkv_out_dim - 2 * kv_head_dim)
        if q_dim <= 0 or q_dim % kv_head_dim != 0:
            raise ValueError(
                f"invalid decoder qkv layout: qkv_out_dim={qkv_out_dim} kv_head_dim={kv_head_dim}"
            )
        num_heads = int(q_dim // kv_head_dim)
        rope = self._gemma_rope_weights(
            seq_len=suffix_len,
            batch_size=bk,
            head_dim=kv_head_dim,
            device=device,
            position_offset=prefix_len,
        )
        if use_fp32_exact:
            rope = rope.to(device=device, dtype=torch.float32)
            prefix_mask = torch.ones((prefix_len,), device=device, dtype=torch.bool)
            suffix_mask = torch.ones((suffix_len,), device=device, dtype=torch.bool)
            verify_mask = torch.ones((suffix_len, prefix_len + suffix_len), device=device, dtype=torch.bool)
            verify_mask[0, prefix_len + 1 :] = False
            scale = float(kv_head_dim) ** -0.5
            for layer_idx in range(decoder_layers):
                x_flat = x.reshape(bk * suffix_len, hidden_size)
                qkv = _matmul_fp32(_rms_norm_fp32(x_flat), weights["decoder_attn_qkv_w"][layer_idx])
                q_flat = qkv[:, :q_dim]
                suffix_k_flat = qkv[:, q_dim : q_dim + kv_head_dim]
                suffix_v_flat = qkv[:, q_dim + kv_head_dim :]
                q_flat = _apply_gemma_rope_fp32(q_flat, rope, head_dim=kv_head_dim)
                suffix_k_flat = _apply_gemma_rope_fp32(suffix_k_flat, rope, head_dim=kv_head_dim)
                q = q_flat.view(bk, suffix_len, num_heads, kv_head_dim).permute(0, 2, 1, 3).contiguous()
                prefix_k = encoder_k[layer_idx].to(dtype=torch.float32)[None, None, :, :].expand(bk, num_heads, -1, -1)
                prefix_v = encoder_v[layer_idx].to(dtype=torch.float32)[None, None, :, :].expand(bk, num_heads, -1, -1)
                suffix_k = suffix_k_flat.view(bk, suffix_len, kv_head_dim)[:, None, :, :].expand(-1, num_heads, -1, -1)
                suffix_v = suffix_v_flat.view(bk, suffix_len, kv_head_dim)[:, None, :, :].expand(-1, num_heads, -1, -1)
                keys = torch.cat((prefix_k, suffix_k), dim=2)
                values = torch.cat((prefix_v, suffix_v), dim=2)
                mask = torch.cat((prefix_mask[None, :].expand(suffix_len, -1), suffix_mask[None, :].expand(suffix_len, -1)), dim=1)
                mask = (mask & verify_mask).view(1, 1, suffix_len, prefix_len + suffix_len)
                logits = torch.matmul(q, keys.transpose(-1, -2)) * scale
                logits = logits.masked_fill(~mask, float("-inf"))
                attn_out = torch.matmul(torch.softmax(logits, dim=-1), values)
                attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(bk * suffix_len, q_dim)
                x = (
                    x_flat
                    + _matmul_fp32(attn_out, weights["decoder_attn_o_w"][layer_idx])
                ).view(bk, suffix_len, hidden_size)
                x_norm = _rms_norm_fp32(x.reshape(bk * suffix_len, hidden_size))
                gate = _matmul_fp32(x_norm, weights["decoder_ffn_gate_w"][layer_idx])
                up = _matmul_fp32(x_norm, weights["decoder_ffn_up_w"][layer_idx])
                ffn_hidden = torch.nn.functional.gelu(gate, approximate="tanh") * up
                x = (
                    x.reshape(bk * suffix_len, hidden_size)
                    + _matmul_fp32(ffn_hidden, weights["decoder_ffn_down_w"][layer_idx])
                ).view(bk, suffix_len, hidden_size)
            delta = _matmul_bias_fp32(
                _rms_norm_fp32(x[:, 1:, :].reshape(bk * action_horizon, hidden_size)),
                weights["decoder_action_fused_out_proj_w"],
                weights["decoder_action_fused_out_proj_b"],
            ).view(bk, action_horizon, action_dim)
            velocity = delta * float(self._triton_delta_to_velocity_scale(weights=weights))
            x0_hat = x_t_bk.to(device=device, dtype=torch.float32) - timestep_bk.to(device=device, dtype=torch.float32)[
                :, None, None
            ] * velocity
        else:
            key_valid = torch.ones((bk, prefix_len + suffix_len), device=device, dtype=torch.bool)
            for layer_idx in range(decoder_layers):
                x_flat = x.reshape(bk * suffix_len, hidden_size)
                q_flat, suffix_k_flat, suffix_v_flat = _rms_qkv_rope_kernel(
                    x_flat,
                    weights["decoder_attn_qkv_w"][layer_idx],
                    rope,
                    num_heads=num_heads,
                    head_dim=kv_head_dim,
                )
                q_rows = (
                    q_flat.view(bk, suffix_len, num_heads, kv_head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .reshape(bk * num_heads * suffix_len, kv_head_dim)
                )
                prefix_k = (
                    encoder_k[layer_idx].unsqueeze(0).expand(bk, -1, -1).contiguous().reshape(bk * prefix_len, kv_head_dim)
                )
                prefix_v = (
                    encoder_v[layer_idx].unsqueeze(0).expand(bk, -1, -1).contiguous().reshape(bk * prefix_len, kv_head_dim)
                )
                suffix_k = suffix_k_flat.view(bk, suffix_len, kv_head_dim).reshape(bk * suffix_len, kv_head_dim)
                suffix_v = suffix_v_flat.view(bk, suffix_len, kv_head_dim).reshape(bk * suffix_len, kv_head_dim)
                attn_rows = self._attention_kernel(
                    q_rows=q_rows,
                    prefix_k=prefix_k,
                    prefix_v=prefix_v,
                    suffix_k=suffix_k,
                    suffix_v=suffix_v,
                    key_valid=key_valid,
                    batch_size=bk,
                    query_len=suffix_len,
                    prefix_len=prefix_len,
                    suffix_len=suffix_len,
                    num_heads=num_heads,
                    head_dim=kv_head_dim,
                    block_suffix_for_query0=True,
                )
                attn_out = (
                    attn_rows.view(bk, num_heads, suffix_len, kv_head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .view(bk * suffix_len, q_dim)
                )
                x = _matmul_residual_kernel(attn_out, weights["decoder_attn_o_w"][layer_idx], x_flat).view(
                    bk, suffix_len, hidden_size
                )
                ffn_hidden = _rms_matmul_gate_kernel(
                    x.reshape(bk * suffix_len, hidden_size),
                    weights["decoder_ffn_gate_w"][layer_idx],
                    weights["decoder_ffn_up_w"][layer_idx],
                )
                x = _matmul_residual_kernel(
                    ffn_hidden,
                    weights["decoder_ffn_down_w"][layer_idx],
                    x.reshape(bk * suffix_len, hidden_size),
                ).view(
                    bk,
                    suffix_len,
                    hidden_size,
                )

            delta = _rms_matmul_bias_kernel(
                x[:, 1:, :].reshape(bk * action_horizon, hidden_size),
                weights["decoder_action_fused_out_proj_w"],
                weights["decoder_action_fused_out_proj_b"],
            ).view(bk, action_horizon, action_dim)
            velocity = delta.to(device=device, dtype=torch.float32) * float(
                self._triton_delta_to_velocity_scale(weights=weights)
            )
            x0_hat = self._x0_hat_from_velocity(
                x_t=x_t_bk.to(device=device, dtype=torch.float32),
                velocity=velocity,
                timestep=timestep_bk.to(device=device, dtype=torch.float32),
            )
        return x0_hat.to(dtype=torch.float32)

    def _run_batched_verify(
        self,
        *,
        full_runtime: Any,
        cache_snapshot: FullCacheSnapshot,
        observation_state_normalized: torch.Tensor,
        x_t_bk: torch.Tensor,
        timestep_bk: torch.Tensor,
    ) -> torch.Tensor:
        weights = getattr(full_runtime, "weights", None)
        if not isinstance(weights, Mapping):
            raise ValueError("full runtime weights are unavailable for verify")
        if self._verify_fast_path_supported(weights=weights, cache_snapshot=cache_snapshot, x_t_bk=x_t_bk):
            if self._verify_fast_graph_enabled():
                try:
                    return self._run_batched_verify_fast_graph(
                        full_runtime=full_runtime,
                        cache_snapshot=cache_snapshot,
                        observation_state_normalized=observation_state_normalized,
                        x_t_bk=x_t_bk,
                        timestep_bk=timestep_bk,
                    )
                except Exception:
                    self._verify_fast_graph = None
                    self._verify_fast_graph_failed = True
            return self._run_batched_verify_fast(
                full_runtime=full_runtime,
                cache_snapshot=cache_snapshot,
                observation_state_normalized=observation_state_normalized,
                x_t_bk=x_t_bk,
                timestep_bk=timestep_bk,
            )
        return self._run_batched_verify_generic(
            full_runtime=full_runtime,
            cache_snapshot=cache_snapshot,
            observation_state_normalized=observation_state_normalized,
            x_t_bk=x_t_bk,
            timestep_bk=timestep_bk,
        )

    def run_verify(
        self,
        cache_snapshot: FullCacheSnapshot,
        observation_images_normalized: torch.Tensor,
        observation_state_normalized: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: Sequence[float],
        *,
        full_runtime: Any | None = None,
    ) -> torch.Tensor:
        del observation_images_normalized
        unavailable_reasons = self._full_verify_unavailable_reasons(
            full_runtime=full_runtime,
            cache_snapshot=cache_snapshot,
        )
        if unavailable_reasons:
            raise RuntimeError(
                "Spec Triton verify cannot run the full verifier; refusing identity verify fallback: "
                + "; ".join(unavailable_reasons)
            )

        x_t_bk, timestep_bk = self._build_verify_batch(noise=noise, x0_draft=x0_draft, t_list=t_list)

        if cache_snapshot.encoder_k is None:
            raise RuntimeError("Spec Triton verify requires encoder_k in the full cache snapshot")
        x0_hat_bk = self._run_batched_verify(
            full_runtime=full_runtime,
            cache_snapshot=cache_snapshot,
            observation_state_normalized=observation_state_normalized,
            x_t_bk=x_t_bk.to(device=cache_snapshot.encoder_k.device, dtype=torch.float32),
            timestep_bk=timestep_bk.to(device=cache_snapshot.encoder_k.device, dtype=torch.float32),
        )

        batch_size = int(x0_draft.shape[0]) if x0_draft.ndim == 3 else 1
        return x0_hat_bk.reshape(batch_size, len(tuple(t_list)), self.chunk_size, 32)

    def _run_verify_postprocess_fused(
        self,
        *,
        x0_hat: torch.Tensor,
        x0_draft: torch.Tensor,
        tau_radius: float,
        dist_dims: int,
        max_exec_steps: int,
        last_gripper: torch.Tensor | None,
        gripper_switch_threshold: float,
        enable_gripper_verify: bool,
        enable_gripper_post_verify: bool,
    ) -> SpecVerifyOutput | None:
        if _spec_triton_env("POSTPROCESS_FUSED", "1") == "0":
            return None
        if x0_hat.device.type != "cuda" or x0_draft.device.type != "cuda":
            return None
        if x0_hat.ndim != 4 or x0_draft.ndim != 3:
            return None
        if not x0_hat.is_contiguous() or not x0_draft.is_contiguous():
            return None

        b, k, h, d = map(int, x0_hat.shape)
        if tuple(x0_draft.shape) != (b, h, d):
            return None
        eval_h = int(min(h, max(1, int(max_exec_steps))))
        eval_d = int(min(d, int(dist_dims)))
        if d >= 7:
            eval_d = int(min(eval_d, 6))
        if eval_d <= 0:
            return None

        buffers = self._ensure_verify_postprocess_buffers(
            batch_size=b,
            horizon=h,
            action_dim=d,
            device=x0_hat.device,
        )
        has_gripper_prev = last_gripper is not None and int(d) >= 7
        gripper_prev = buffers.metrics
        if has_gripper_prev:
            if last_gripper is None or last_gripper.ndim != 1 or int(last_gripper.shape[0]) != b:
                return None
            gripper_prev = last_gripper.to(device=x0_hat.device, dtype=torch.float32).contiguous()

        _spec_verify_accept_metrics_kernel[(1,)](
            x0_hat.view(b * k, h, d),
            x0_draft,
            gripper_prev,
            buffers.accepted_prefix_len,
            buffers.action_prefix_len,
            buffers.gripper_verify_stop_mask,
            buffers.gripper_switch_cut_mask,
            buffers.metrics,
            batch_size=b,
            verify_k=k,
            horizon=h,
            action_dim=d,
            eval_h=eval_h,
            eval_d=eval_d,
            tau_radius=float(tau_radius),
            gripper_switch_threshold=float(gripper_switch_threshold),
            has_gripper_prev=bool(has_gripper_prev),
            enable_gripper_verify=bool(enable_gripper_verify),
            enable_gripper_post_verify=bool(enable_gripper_post_verify),
        )
        _spec_verify_stitch_kernel[(max(1, triton.cdiv(b * h * d, 256)),)](
            x0_hat.view(b * k, h, d),
            x0_draft,
            buffers.actions,
            buffers.action_prefix_len,
            buffers.gripper_verify_stop_mask,
            total_values=b * h * d,
            verify_k=k,
            horizon=h,
            action_dim=d,
            BLOCK_SIZE=256,
        )
        return SpecVerifyOutput(
            x0_hat=x0_hat,
            actions=buffers.actions,
            metrics=buffers.metrics,
            accepted_prefix_len=buffers.accepted_prefix_len,
        )

    def _run_verify_semantics_fast_no_graph(
        self,
        *,
        cache_snapshot: FullCacheSnapshot,
        observation_images_normalized: torch.Tensor,
        observation_state_normalized: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: Sequence[float],
        tau_radius: float,
        dist_dims: int,
        max_exec_steps: int,
        last_gripper: torch.Tensor | None,
        gripper_switch_threshold: float,
        enable_gripper_verify: bool,
        enable_gripper_post_verify: bool,
        full_runtime: Any | None = None,
    ) -> SpecVerifyOutput | None:
        del observation_images_normalized
        if x0_draft.ndim != 3 or noise.ndim != 3:
            return None
        unavailable_reasons = self._full_verify_unavailable_reasons(
            full_runtime=full_runtime,
            cache_snapshot=cache_snapshot,
        )
        if unavailable_reasons:
            return None
        weights = getattr(full_runtime, "weights", None)
        if not isinstance(weights, Mapping):
            return None
        x_t_bk, timestep_bk = self._build_verify_batch(noise=noise, x0_draft=x0_draft, t_list=t_list)
        if not self._verify_fast_path_supported(weights=weights, cache_snapshot=cache_snapshot, x_t_bk=x_t_bk):
            return None
        if cache_snapshot.encoder_k is None:
            return None
        x0_hat_bk = self._run_batched_verify_fast(
            full_runtime=full_runtime,
            cache_snapshot=cache_snapshot,
            observation_state_normalized=observation_state_normalized,
            x_t_bk=x_t_bk.to(device=cache_snapshot.encoder_k.device, dtype=torch.float32),
            timestep_bk=timestep_bk.to(device=cache_snapshot.encoder_k.device, dtype=torch.float32),
        )
        b, h, d = map(int, x0_draft.shape)
        x0_hat = x0_hat_bk.reshape(b, len(tuple(t_list)), h, d)
        return self._run_verify_postprocess_fused(
            x0_hat=x0_hat,
            x0_draft=x0_draft,
            tau_radius=tau_radius,
            dist_dims=dist_dims,
            max_exec_steps=max_exec_steps,
            last_gripper=last_gripper,
            gripper_switch_threshold=gripper_switch_threshold,
            enable_gripper_verify=enable_gripper_verify,
            enable_gripper_post_verify=enable_gripper_post_verify,
        )

    def _verify_semantics_graph_enabled(self) -> bool:
        return (
            self.device.type == "cuda"
            and torch.cuda.is_available()
            and _spec_triton_env("VERIFY_GRAPH", "1") != "0"
            and _spec_triton_env("VERIFY_SEMANTICS_GRAPH", "1") != "0"
            and _spec_triton_env("POSTPROCESS_FUSED", "1") != "0"
            and _spec_triton_env("VERIFY_EXACT_INPUT", "0") != "1"
            and _spec_triton_env("VERIFY_EXACT_FP32", "0") != "1"
            and not self._verify_semantics_graph_failed
        )

    def _verify_semantics_graph_key(
        self,
        *,
        weights: Mapping[str, torch.Tensor],
        cache_snapshot: FullCacheSnapshot,
        observation_state_normalized: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: Sequence[float],
        tau_radius: float,
        dist_dims: int,
        max_exec_steps: int,
        last_gripper: torch.Tensor | None,
        gripper_switch_threshold: float,
        enable_gripper_verify: bool,
        enable_gripper_post_verify: bool,
    ) -> tuple[Any, ...]:
        if cache_snapshot.encoder_k is None or cache_snapshot.encoder_v is None:
            raise ValueError("full cache snapshot is missing encoder KV tensors")
        return (
            id(weights),
            str(x0_draft.device),
            self.compute_dtype,
            int(cache_snapshot.encoder_seq_len),
            tuple(cache_snapshot.encoder_k.shape),
            tuple(cache_snapshot.encoder_v.shape),
            tuple(observation_state_normalized.shape),
            tuple(noise.shape),
            tuple(x0_draft.shape),
            tuple(float(t) for t in t_list),
            float(tau_radius),
            int(dist_dims),
            int(max_exec_steps),
            float(gripper_switch_threshold),
            bool(enable_gripper_verify),
            bool(enable_gripper_post_verify),
            last_gripper is not None,
            tuple(last_gripper.shape) if last_gripper is not None else None,
            tuple(weights["decoder_attn_qkv_w"].shape),
        )

    @staticmethod
    def _copy_verify_semantics_graph_inputs(
        graph_state: _VerifySemanticsGraph,
        *,
        cache_snapshot: FullCacheSnapshot,
        observation_state_normalized: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        last_gripper: torch.Tensor | None,
    ) -> None:
        if cache_snapshot.encoder_k is None or cache_snapshot.encoder_v is None:
            raise ValueError("full cache snapshot is missing encoder KV tensors")
        graph_state.noise.copy_(noise.to(device=graph_state.noise.device, dtype=graph_state.noise.dtype))
        graph_state.x0_draft.copy_(
            x0_draft.to(device=graph_state.x0_draft.device, dtype=graph_state.x0_draft.dtype)
        )
        graph_state.state.copy_(
            observation_state_normalized.to(device=graph_state.state.device, dtype=graph_state.state.dtype)
        )
        if graph_state.last_gripper is not None:
            if last_gripper is None:
                raise ValueError("verify semantics graph expects last_gripper")
            graph_state.last_gripper.copy_(
                last_gripper.to(device=graph_state.last_gripper.device, dtype=graph_state.last_gripper.dtype)
            )
        encoder_k_source = (id(cache_snapshot.encoder_k), int(cache_snapshot.encoder_k.data_ptr()))
        encoder_v_source = (id(cache_snapshot.encoder_v), int(cache_snapshot.encoder_v.data_ptr()))
        if graph_state.encoder_k_source != encoder_k_source:
            graph_state.encoder_k.copy_(
                cache_snapshot.encoder_k.to(device=graph_state.encoder_k.device, dtype=graph_state.encoder_k.dtype)
            )
            graph_state.encoder_k_source = encoder_k_source
        if graph_state.encoder_v_source != encoder_v_source:
            graph_state.encoder_v.copy_(
                cache_snapshot.encoder_v.to(device=graph_state.encoder_v.device, dtype=graph_state.encoder_v.dtype)
            )
            graph_state.encoder_v_source = encoder_v_source

    def _get_or_create_verify_semantics_graph(
        self,
        *,
        cache_snapshot: FullCacheSnapshot,
        observation_images_normalized: torch.Tensor,
        observation_state_normalized: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: Sequence[float],
        tau_radius: float,
        dist_dims: int,
        max_exec_steps: int,
        last_gripper: torch.Tensor | None,
        gripper_switch_threshold: float,
        enable_gripper_verify: bool,
        enable_gripper_post_verify: bool,
        full_runtime: Any | None = None,
    ) -> _VerifySemanticsGraph:
        weights = getattr(full_runtime, "weights", None)
        if not isinstance(weights, Mapping):
            raise ValueError("full runtime weights are unavailable for verify")
        if cache_snapshot.encoder_k is None or cache_snapshot.encoder_v is None:
            raise ValueError("full cache snapshot is missing encoder KV tensors")
        key = self._verify_semantics_graph_key(
            weights=weights,
            cache_snapshot=cache_snapshot,
            observation_state_normalized=observation_state_normalized,
            noise=noise,
            x0_draft=x0_draft,
            t_list=t_list,
            tau_radius=tau_radius,
            dist_dims=dist_dims,
            max_exec_steps=max_exec_steps,
            last_gripper=last_gripper,
            gripper_switch_threshold=gripper_switch_threshold,
            enable_gripper_verify=enable_gripper_verify,
            enable_gripper_post_verify=enable_gripper_post_verify,
        )
        graph_state = self._verify_semantics_graph
        if graph_state is not None and graph_state.key == key:
            return graph_state

        device = x0_draft.device
        static_noise = torch.empty_like(noise.to(device=device, dtype=torch.float32), device=device, dtype=torch.float32)
        static_x0_draft = torch.empty_like(
            x0_draft.to(device=device, dtype=torch.float32),
            device=device,
            dtype=torch.float32,
        )
        static_state = torch.empty_like(
            observation_state_normalized.to(device=device, dtype=torch.float32),
            device=device,
            dtype=torch.float32,
        )
        static_last_gripper = None
        if last_gripper is not None:
            static_last_gripper = torch.empty_like(
                last_gripper.to(device=device, dtype=torch.float32),
                device=device,
                dtype=torch.float32,
            )
        static_encoder_k = torch.empty_like(cache_snapshot.encoder_k, device=device, dtype=self.compute_dtype)
        static_encoder_v = torch.empty_like(cache_snapshot.encoder_v, device=device, dtype=self.compute_dtype)
        static_cache = FullCacheSnapshot(
            encoder_seq_len=int(cache_snapshot.encoder_seq_len),
            encoder_x=None,
            encoder_k=static_encoder_k,
            encoder_v=static_encoder_v,
        )
        static_runtime = _VerifyGraphRuntime(weights=weights)
        graph_state = _VerifySemanticsGraph(
            key=key,
            graph=torch.cuda.CUDAGraph(),
            noise=static_noise,
            x0_draft=static_x0_draft,
            state=static_state,
            last_gripper=static_last_gripper,
            encoder_k=static_encoder_k,
            encoder_v=static_encoder_v,
            cache_snapshot=static_cache,
            full_runtime=static_runtime,
            output=SpecVerifyOutput(
                x0_hat=static_x0_draft,
                actions=static_x0_draft,
                metrics=torch.empty((5,), device=device, dtype=torch.float32),
                accepted_prefix_len=torch.empty((int(x0_draft.shape[0]),), device=device, dtype=torch.int64),
            ),
        )
        self._copy_verify_semantics_graph_inputs(
            graph_state,
            cache_snapshot=cache_snapshot,
            observation_state_normalized=observation_state_normalized,
            noise=noise,
            x0_draft=x0_draft,
            last_gripper=last_gripper,
        )

        for _ in range(2):
            output = self._run_verify_semantics_fast_no_graph(
                cache_snapshot=graph_state.cache_snapshot,
                observation_images_normalized=observation_images_normalized,
                observation_state_normalized=graph_state.state,
                noise=graph_state.noise,
                x0_draft=graph_state.x0_draft,
                t_list=t_list,
                tau_radius=tau_radius,
                dist_dims=dist_dims,
                max_exec_steps=max_exec_steps,
                last_gripper=graph_state.last_gripper,
                gripper_switch_threshold=gripper_switch_threshold,
                enable_gripper_verify=enable_gripper_verify,
                enable_gripper_post_verify=enable_gripper_post_verify,
                full_runtime=graph_state.full_runtime,
            )
            if output is None:
                raise RuntimeError("verify semantics graph requires the fast verifier and fused postprocess")
            graph_state.output = output
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph_state.graph):
            output = self._run_verify_semantics_fast_no_graph(
                cache_snapshot=graph_state.cache_snapshot,
                observation_images_normalized=observation_images_normalized,
                observation_state_normalized=graph_state.state,
                noise=graph_state.noise,
                x0_draft=graph_state.x0_draft,
                t_list=t_list,
                tau_radius=tau_radius,
                dist_dims=dist_dims,
                max_exec_steps=max_exec_steps,
                last_gripper=graph_state.last_gripper,
                gripper_switch_threshold=gripper_switch_threshold,
                enable_gripper_verify=enable_gripper_verify,
                enable_gripper_post_verify=enable_gripper_post_verify,
                full_runtime=graph_state.full_runtime,
            )
            if output is None:
                raise RuntimeError("verify semantics graph requires the fast verifier and fused postprocess")
            graph_state.output = output
        torch.cuda.synchronize(device)
        self._verify_semantics_graph = graph_state
        return graph_state

    def _run_verify_semantics_fast_graph(
        self,
        *,
        cache_snapshot: FullCacheSnapshot,
        observation_images_normalized: torch.Tensor,
        observation_state_normalized: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: Sequence[float],
        tau_radius: float,
        dist_dims: int,
        max_exec_steps: int,
        last_gripper: torch.Tensor | None,
        gripper_switch_threshold: float,
        enable_gripper_verify: bool,
        enable_gripper_post_verify: bool,
        full_runtime: Any | None = None,
    ) -> SpecVerifyOutput | None:
        if not self._verify_semantics_graph_enabled():
            return None
        if x0_draft.ndim != 3 or noise.ndim != 3 or x0_draft.device.type != "cuda":
            return None
        try:
            graph_state = self._get_or_create_verify_semantics_graph(
                cache_snapshot=cache_snapshot,
                observation_images_normalized=observation_images_normalized,
                observation_state_normalized=observation_state_normalized,
                noise=noise,
                x0_draft=x0_draft,
                t_list=t_list,
                tau_radius=tau_radius,
                dist_dims=dist_dims,
                max_exec_steps=max_exec_steps,
                last_gripper=last_gripper,
                gripper_switch_threshold=gripper_switch_threshold,
                enable_gripper_verify=enable_gripper_verify,
                enable_gripper_post_verify=enable_gripper_post_verify,
                full_runtime=full_runtime,
            )
            self._copy_verify_semantics_graph_inputs(
                graph_state,
                cache_snapshot=cache_snapshot,
                observation_state_normalized=observation_state_normalized,
                noise=noise,
                x0_draft=x0_draft,
                last_gripper=last_gripper,
            )
            graph_state.graph.replay()
            return graph_state.output
        except Exception:
            self._verify_semantics_graph = None
            self._verify_semantics_graph_failed = True
            return None

    def run_verify_semantics(
        self,
        cache_snapshot: FullCacheSnapshot,
        observation_images_normalized: torch.Tensor,
        observation_state_normalized: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: Sequence[float],
        *,
        tau_radius: float,
        dist_dims: int,
        max_exec_steps: int,
        last_gripper: torch.Tensor | None,
        gripper_switch_threshold: float,
        enable_gripper_verify: bool,
        enable_gripper_post_verify: bool,
        full_runtime: Any | None = None,
    ) -> SpecVerifyOutput:
        x0_hat = self.run_verify(
            cache_snapshot,
            observation_images_normalized,
            observation_state_normalized,
            noise,
            x0_draft,
            t_list,
            full_runtime=full_runtime,
        )
        fused = self._run_verify_postprocess_fused(
            x0_hat=x0_hat,
            x0_draft=x0_draft,
            tau_radius=tau_radius,
            dist_dims=dist_dims,
            max_exec_steps=max_exec_steps,
            last_gripper=last_gripper,
            gripper_switch_threshold=gripper_switch_threshold,
            enable_gripper_verify=enable_gripper_verify,
            enable_gripper_post_verify=enable_gripper_post_verify,
        )
        if fused is not None:
            return fused

        eval_h = int(min(int(x0_draft.shape[1]), max(1, int(max_exec_steps))))
        accepted_prefix_len, dist = _compute_radius_prefix_acceptance(
            x0_draft=x0_draft,
            x0_hat=x0_hat,
            tau_radius=float(tau_radius),
            dist_dims=int(dist_dims),
            eval_h=int(eval_h),
        )
        x0_tail = x0_hat.mean(dim=1)
        x0_out = _stitch_radius_prefix_output(
            x0_draft=x0_draft,
            x0_tail=x0_tail,
            accepted_prefix_len=accepted_prefix_len,
        )
        batch_size = int(x0_out.shape[0])
        gripper_switch_cut_mask = torch.zeros((batch_size,), device=x0_out.device, dtype=torch.bool)
        gripper_verify_stop_mask = torch.zeros((batch_size,), device=x0_out.device, dtype=torch.bool)
        if bool(enable_gripper_verify):
            gripper_verify_stop_mask = _detect_verify_gripper_switch_any_k(
                x0_hat=x0_hat,
                gripper_prev=last_gripper,
                gripper_switch_threshold=float(gripper_switch_threshold),
                eval_h=int(eval_h),
            )
            accepted_prefix_len = torch.where(
                gripper_verify_stop_mask,
                torch.zeros_like(accepted_prefix_len),
                accepted_prefix_len,
            )
            x0_out = torch.where(gripper_verify_stop_mask[:, None, None], x0_tail, x0_out)
        if bool(enable_gripper_post_verify):
            accepted_after_cut, gripper_switch_cut_mask = _truncate_accepted_prefix_on_gripper_switch(
                x0_out=x0_out,
                accepted_prefix_len=accepted_prefix_len,
                gripper_prev=last_gripper,
                gripper_switch_threshold=float(gripper_switch_threshold),
            )
            accepted_prefix_len = torch.where(gripper_verify_stop_mask, accepted_prefix_len, accepted_after_cut)
            gripper_switch_cut_mask = gripper_switch_cut_mask & (~gripper_verify_stop_mask)

        metrics = _make_speculative_metrics(
            radius_dist=dist.mean(),
            accepted_prefix_len_mean=accepted_prefix_len.to(dtype=torch.float32).mean(),
            gripper_switch_cut_rate=gripper_switch_cut_mask.to(dtype=torch.float32).mean(),
            scheduled_full_fallback_gripper=(gripper_switch_cut_mask | gripper_verify_stop_mask)
            .any()
            .to(dtype=torch.float32),
            gripper_verify_stop_rate=gripper_verify_stop_mask.to(dtype=torch.float32).mean(),
        )
        return SpecVerifyOutput(
            x0_hat=x0_hat,
            actions=x0_out,
            metrics=metrics,
            accepted_prefix_len=accepted_prefix_len,
        )

    def run_verify_with_timing(
        self,
        cache_snapshot: FullCacheSnapshot,
        observation_images_normalized: torch.Tensor,
        observation_state_normalized: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: Sequence[float],
        *,
        full_runtime: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        timing_device = self.device
        if cache_snapshot.encoder_k is not None:
            timing_device = cache_snapshot.encoder_k.device
        x0_hat, verify_ms = _time_ms(
            lambda: self.run_verify(
                cache_snapshot,
                observation_images_normalized,
                observation_state_normalized,
                noise,
                x0_draft,
                t_list,
                full_runtime=full_runtime,
            ),
            device=timing_device,
        )
        return x0_hat, {
            "action_verify_ms": float(verify_ms),
        }

    def run_verify_semantics_with_timing(
        self,
        cache_snapshot: FullCacheSnapshot,
        observation_images_normalized: torch.Tensor,
        observation_state_normalized: torch.Tensor,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: Sequence[float],
        *,
        tau_radius: float,
        dist_dims: int,
        max_exec_steps: int,
        last_gripper: torch.Tensor | None,
        gripper_switch_threshold: float,
        enable_gripper_verify: bool,
        enable_gripper_post_verify: bool,
        full_runtime: Any | None = None,
    ) -> tuple[SpecVerifyOutput, dict[str, float]]:
        timing_device = self.device
        if cache_snapshot.encoder_k is not None:
            timing_device = cache_snapshot.encoder_k.device

        def _run_semantics() -> SpecVerifyOutput:
            graphed = self._run_verify_semantics_fast_graph(
                cache_snapshot=cache_snapshot,
                observation_images_normalized=observation_images_normalized,
                observation_state_normalized=observation_state_normalized,
                noise=noise,
                x0_draft=x0_draft,
                t_list=t_list,
                tau_radius=tau_radius,
                dist_dims=dist_dims,
                max_exec_steps=max_exec_steps,
                last_gripper=last_gripper,
                gripper_switch_threshold=gripper_switch_threshold,
                enable_gripper_verify=enable_gripper_verify,
                enable_gripper_post_verify=enable_gripper_post_verify,
                full_runtime=full_runtime,
            )
            if graphed is not None:
                return graphed
            return self.run_verify_semantics(
                cache_snapshot,
                observation_images_normalized,
                observation_state_normalized,
                noise,
                x0_draft,
                t_list,
                tau_radius=tau_radius,
                dist_dims=dist_dims,
                max_exec_steps=max_exec_steps,
                last_gripper=last_gripper,
                gripper_switch_threshold=gripper_switch_threshold,
                enable_gripper_verify=enable_gripper_verify,
                enable_gripper_post_verify=enable_gripper_post_verify,
                full_runtime=full_runtime,
            )

        result, verify_ms = _time_ms(
            _run_semantics,
            device=timing_device,
        )
        return result, {
            "action_verify_ms": float(verify_ms),
        }
