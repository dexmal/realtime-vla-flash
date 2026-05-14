import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


_PI0_HIDDEN_SIZE = 2048
_PI0_ACTION_DIM = 32


def _prefer_local_repo_src() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    for path in (str(repo_root), str(src_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)


_prefer_local_repo_src()

from scripts.spec.triton import triton_pi0_runtime as triton_runtime  # noqa: E402


def _sync() -> None:
    torch.cuda.synchronize()


def _random_micro_checkpoint(*, prompt_len: int) -> dict[str, torch.Tensor]:
    checkpoint = {
        "language_embeds": torch.randn((int(prompt_len), _PI0_HIDDEN_SIZE), dtype=torch.bfloat16),
    }
    # Keep the exact verify input path enabled without loading model weights.
    checkpoint.update(
        {
            "decoder_action_in_proj_w": torch.zeros((32, 1024), dtype=torch.bfloat16),
            "decoder_action_in_proj_b": torch.zeros((1024,), dtype=torch.bfloat16),
            "decoder_action_time_mlp_in_w": torch.zeros((2048, 1024), dtype=torch.bfloat16),
            "decoder_action_time_mlp_in_b": torch.zeros((1024,), dtype=torch.bfloat16),
        }
    )
    return checkpoint


def _random_draft_checkpoint(
    *,
    chunk_size: int,
    out_dim: int,
    ffn_hidden_size: int,
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor | dict[str, int]]:
    qkv_dim = (int(num_heads) + 2) * int(head_dim)
    return {
        "meta": {
            "img_dim": _PI0_HIDDEN_SIZE,
            "chunk_m": int(chunk_size),
            "out_dim": int(out_dim),
            "draft_num_heads": int(num_heads),
            "draft_num_kv_heads": 1,
            "draft_head_dim": int(head_dim),
        },
        "draft_state_in_proj_w": torch.zeros((_PI0_HIDDEN_SIZE, 32), dtype=torch.bfloat16),
        "draft_state_in_proj_b": torch.zeros((_PI0_HIDDEN_SIZE,), dtype=torch.bfloat16),
        "draft_action_queries": torch.zeros((int(chunk_size), _PI0_HIDDEN_SIZE), dtype=torch.bfloat16),
        "draft_qkv_w": torch.zeros((qkv_dim, _PI0_HIDDEN_SIZE), dtype=torch.bfloat16),
        "draft_attn_o_w": torch.zeros((_PI0_HIDDEN_SIZE, _PI0_HIDDEN_SIZE), dtype=torch.bfloat16),
        "draft_ffn_gate_w": torch.zeros((int(ffn_hidden_size), _PI0_HIDDEN_SIZE), dtype=torch.bfloat16),
        "draft_ffn_up_w": torch.zeros((int(ffn_hidden_size), _PI0_HIDDEN_SIZE), dtype=torch.bfloat16),
        "draft_ffn_down_w": torch.zeros((_PI0_HIDDEN_SIZE, int(ffn_hidden_size)), dtype=torch.bfloat16),
        "draft_input_layernorm_w": torch.zeros((_PI0_HIDDEN_SIZE,), dtype=torch.bfloat16),
        "draft_post_attention_layernorm_w": torch.zeros((_PI0_HIDDEN_SIZE,), dtype=torch.bfloat16),
        "draft_action_head_w": torch.zeros((int(out_dim), _PI0_HIDDEN_SIZE), dtype=torch.bfloat16),
        "draft_action_head_b": torch.zeros((int(out_dim),), dtype=torch.bfloat16),
    }


def _summarize(values: list[float]) -> str:
    return (
        f"mean={statistics.fmean(values):.3f} ms, "
        f"median={statistics.median(values):.3f} ms, "
        f"min={min(values):.3f} ms, max={max(values):.3f} ms"
    )


def _run_once(
    *,
    session: triton_runtime.SpecTritonRuntimeSession,
    prepared: triton_runtime.TritonPreparedObservation,
    cache_snapshot,
    noise: torch.Tensor,
    t_list: tuple[float, ...],
    tau_radius: float,
    dist_dims: int,
    max_exec_steps: int,
) -> dict[str, float]:
    _sync()
    t0 = time.perf_counter()
    x0_draft, draft_timing = session.run_draft_with_timing(prepared=prepared)
    _, verify_timing = session.run_verify_semantics_with_timing(
        cache_snapshot=cache_snapshot,
        prepared=prepared,
        noise=noise,
        x0_draft=x0_draft,
        t_list=t_list,
        tau_radius=float(tau_radius),
        dist_dims=int(dist_dims),
        max_exec_steps=int(max_exec_steps),
        last_gripper=None,
        gripper_switch_threshold=0.0,
        enable_gripper_verify=True,
        enable_gripper_post_verify=True,
    )
    _sync()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    encoder_ms = float(draft_timing.get("encoder_ms", 0.0))
    draft_ms = float(draft_timing.get("draft_ms", 0.0))
    verify_ms = float(verify_timing.get("action_verify_ms", 0.0))
    return {
        "encoder_ms": encoder_ms,
        "draft_ms": draft_ms,
        "verify_ms": verify_ms,
        "stage_total_ms": encoder_ms + draft_ms + verify_ms,
        "wall_ms": wall_ms,
    }


def benchmark_spec_triton(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Triton Spec benchmark requires CUDA.")
    if int(args.runs) <= 0:
        raise ValueError("--runs must be positive.")
    if int(args.warmup) < 0:
        raise ValueError("--warmup must be non-negative.")
    if int(args.draft_out_dim) <= 0 or int(args.draft_out_dim) > _PI0_ACTION_DIM:
        raise ValueError(f"--draft-out-dim must be in [1, {_PI0_ACTION_DIM}].")
    if int(args.draft_num_heads) * int(args.draft_head_dim) != _PI0_HIDDEN_SIZE:
        raise ValueError(
            "--draft-num-heads * --draft-head-dim must equal the fixed pi0 hidden size "
            f"{_PI0_HIDDEN_SIZE}."
        )

    torch.manual_seed(int(args.seed))
    checkpoint = _random_micro_checkpoint(prompt_len=int(args.prompt_len))
    draft_checkpoint = _random_draft_checkpoint(
        chunk_size=int(args.chunk_size),
        out_dim=int(args.draft_out_dim),
        ffn_hidden_size=int(args.draft_ffn_hidden_size),
        num_heads=int(args.draft_num_heads),
        head_dim=int(args.draft_head_dim),
    )
    runtime = triton_runtime.create_pi0_inference(
        checkpoint=checkpoint,
        num_views=int(args.num_views),
        chunk_size=int(args.chunk_size),
    )
    draft_runtime = triton_runtime.create_pi0_spec_inference(
        checkpoint=checkpoint,
        draft_checkpoint=draft_checkpoint,
        num_views=int(args.num_views),
        chunk_size=int(args.chunk_size),
    )
    session = triton_runtime.SpecTritonRuntimeSession(
        prompt="<micro-benchmark>",
        runtime=runtime,
        draft_runtime=draft_runtime,
    )

    images = torch.randn(
        (int(args.num_views), 224, 224, 3),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state = torch.randn((32,), device="cuda", dtype=torch.bfloat16)
    noise = torch.randn(
        (int(args.chunk_size), _PI0_ACTION_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    prepared = session.prepare_observation(images=images, state=state)

    _, full_timing = session.run_full_with_timing(prepared=prepared, noise=noise)
    cache_snapshot = session.capture_full_cache_snapshot()

    t_list = tuple(float(x) for x in args.t_list)
    for _ in range(int(args.warmup)):
        _run_once(
            session=session,
            prepared=prepared,
            cache_snapshot=cache_snapshot,
            noise=noise,
            t_list=t_list,
            tau_radius=float(args.tau_radius),
            dist_dims=int(args.dist_dims),
            max_exec_steps=int(args.max_exec_steps),
        )

    records = [
        _run_once(
            session=session,
            prepared=prepared,
            cache_snapshot=cache_snapshot,
            noise=noise,
            t_list=t_list,
            tau_radius=float(args.tau_radius),
            dist_dims=int(args.dist_dims),
            max_exec_steps=int(args.max_exec_steps),
        )
        for _ in range(int(args.runs))
    ]

    print(
        "[Pi0 Spec Triton micro-benchmark]: "
        f"views={args.num_views}, prompt_len={args.prompt_len}, chunk_size={args.chunk_size}, "
        f"draft_out_dim={args.draft_out_dim}, verify_steps={len(t_list)}, "
        f"warmup={args.warmup}, runs={args.runs}"
    )
    if full_timing:
        print(
            "full cache pass: "
            f"encoder={full_timing.get('encoder_ms', 0.0):.3f} ms, "
            f"prefill={full_timing.get('vlm_prefill_ms', 0.0):.3f} ms, "
            f"decoder={full_timing.get('decoder_ms', 0.0):.3f} ms, "
            f"total={full_timing.get('total_ms', 0.0):.3f} ms"
        )
    for key, label in (
        ("encoder_ms", "encoder"),
        ("draft_ms", "draft"),
        ("verify_ms", "verify"),
        ("stage_total_ms", "stage total"),
        ("wall_ms", "wall total"),
    ):
        print(f"{label}: {_summarize([record[key] for record in records])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the no-checkpoint Triton Spec path.")
    parser.add_argument("--num_views", "--num-views", type=int, default=2)
    parser.add_argument("--prompt_len", "--prompt-len", type=int, default=0)
    parser.add_argument("--chunk_size", "--chunk-size", type=int, default=50)
    parser.add_argument("--draft_out_dim", "--draft-out-dim", type=int, default=7)
    parser.add_argument("--draft_ffn_hidden_size", "--draft-ffn-hidden-size", type=int, default=4096)
    parser.add_argument("--draft_num_heads", "--draft-num-heads", type=int, default=8)
    parser.add_argument("--draft_head_dim", "--draft-head-dim", type=int, default=256)
    parser.add_argument("--t_list", "--t-list", type=float, nargs="+", default=(0.10, 0.05))
    parser.add_argument("--tau_radius", "--tau-radius", type=float, default=0.3)
    parser.add_argument("--dist_dims", "--dist-dims", type=int, default=7)
    parser.add_argument("--max_exec_steps", "--max-exec-steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    benchmark_spec_triton(args)


if __name__ == "__main__":
    main()
