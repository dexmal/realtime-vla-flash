from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.spec.triton import triton_pi0_runtime as triton_runtime


def _write_small_spec_session_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    base_weights_path = tmp_path / "base_weights.pkl"
    with base_weights_path.open("wb") as handle:
        triton_runtime.pickle.dump({"shared_weight": torch.tensor([1], dtype=torch.bfloat16)}, handle)

    prompt_dir = tmp_path / "language_embeds"
    prompt_dir.mkdir()
    prompt_path = prompt_dir / "task_a.pt"
    torch.save(torch.ones((2, 8), dtype=torch.float32), prompt_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "prompts": {
                    "task a": {"embed_path": str(prompt_path.relative_to(tmp_path)), "prompt_len": 2},
                }
            }
        ),
        encoding="utf-8",
    )

    draft_ckpt_path = tmp_path / "draft_head.pt"
    torch.save(
        {
            "meta": {
                "img_dim": 8,
                "chunk_m": 4,
                "out_dim": 7,
                "draft_num_heads": 2,
                "draft_num_kv_heads": 1,
                "draft_head_dim": 4,
            },
            "draft_head": {
                "_state_token.weight": torch.zeros((8, 32), dtype=torch.float32),
                "_state_token.bias": torch.zeros((8,), dtype=torch.float32),
                "_action_queries.weight": torch.ones((4, 8), dtype=torch.float32),
                "_gemma_block.self_attn.q_proj.weight": torch.eye(8, dtype=torch.float32),
                "_gemma_block.self_attn.k_proj.weight": torch.ones((4, 8), dtype=torch.float32),
                "_gemma_block.self_attn.v_proj.weight": torch.ones((4, 8), dtype=torch.float32),
                "_gemma_block.self_attn.o_proj.weight": torch.eye(8, dtype=torch.float32),
                "_gemma_block.mlp.gate_proj.weight": torch.ones((16, 8), dtype=torch.float32),
                "_gemma_block.mlp.up_proj.weight": torch.ones((16, 8), dtype=torch.float32),
                "_gemma_block.mlp.down_proj.weight": torch.ones((8, 16), dtype=torch.float32),
                "_gemma_block.input_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
                "_gemma_block.post_attention_layernorm.weight": torch.ones((8,), dtype=torch.bfloat16),
                "_action_head.weight": torch.ones((7, 8), dtype=torch.float32),
                "_action_head.bias": torch.zeros((7,), dtype=torch.float32),
            },
        },
        draft_ckpt_path,
    )
    draft_triton_path = triton_runtime.convert_spec_draft_checkpoint(
        draft_checkpoint_path=draft_ckpt_path,
        output_path=tmp_path / "draft_triton.pkl",
    )
    return base_weights_path, manifest_path, draft_triton_path


def _build_spec_session(tmp_path: Path):
    base_weights_path, manifest_path, draft_triton_path = _write_small_spec_session_artifacts(tmp_path)

    class FakeRuntime:
        def __init__(self, prompt_len: int):
            self.weights = {"language_embeds": torch.zeros((prompt_len, 8), dtype=torch.float32)}

        def forward(self, images, state, noise):
            del images, state
            return noise + 1

        def run_full_with_timing(self, images, state, noise):
            del images, state
            return noise + 1, {
                "encoder_ms": 1.0,
                "vlm_prefill_ms": 2.0,
                "decoder_ms": 3.0,
                "total_ms": 6.0,
            }

    pool = triton_runtime.SpecTritonRuntimePool(
        base_weights_path=base_weights_path,
        manifest_path=manifest_path,
        draft_checkpoint_path=draft_triton_path,
        num_views=2,
        chunk_size=50,
        runtime_factory=lambda *, checkpoint, **_kwargs: FakeRuntime(int(checkpoint["language_embeds"].shape[0])),
    )
    return pool.start_session("task a")


def test_spec_runtime_pool_uses_manifest_built_from_shared_base_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_weights_path = tmp_path / "base_triton.pkl"
    with base_weights_path.open("wb") as handle:
        triton_runtime.pickle.dump(
            {
                "shared_weight": torch.tensor([1], dtype=torch.bfloat16),
                "language_embedding_weight": torch.zeros((16, 8), dtype=torch.bfloat16),
            },
            handle,
        )

    draft_path = tmp_path / "draft_triton.pkl"
    with draft_path.open("wb") as handle:
        triton_runtime.pickle.dump({"meta": {}}, handle)

    observed: dict[str, object] = {}

    def fake_prepare_language_embeds(**kwargs):
        observed["prompt"] = kwargs["prompt"]
        observed["embedding_shape"] = tuple(kwargs["embedding_weight"].shape)
        return torch.ones((3, 8), dtype=torch.bfloat16)

    class FakeRuntime:
        def __init__(self, checkpoint):
            self.weights = {"language_embeds": torch.zeros_like(checkpoint["language_embeds"])}

    monkeypatch.setattr(triton_runtime, "prepare_language_embeds", fake_prepare_language_embeds)
    cache_artifacts = triton_runtime.build_prompt_cache_from_base(
        base_weights_path=base_weights_path,
        cache_dir=tmp_path / "draft_artifact",
        prompts=["task a"],
    )

    pool = triton_runtime.SpecTritonRuntimePool(
        base_weights_path=base_weights_path,
        manifest_path=cache_artifacts["manifest_path"],
        draft_checkpoint_path=draft_path,
        num_views=2,
        chunk_size=50,
        runtime_factory=lambda *, checkpoint, **_kwargs: FakeRuntime(checkpoint),
        spec_runtime_factory=lambda *, checkpoint, **_kwargs: FakeRuntime(checkpoint),
    )

    session = pool.start_session("task a")

    assert observed == {"prompt": "task a", "embedding_shape": (16, 8)}
    assert (tmp_path / "draft_artifact" / "manifest.json").is_file()
    assert (tmp_path / "draft_artifact" / "language_embeds").is_dir()
    assert tuple(session._runtime.weights["language_embeds"].shape) == (3, 8)
    assert tuple(session.draft_runtime.weights["language_embeds"].shape) == (3, 8)



def _load_spec_verify_runtime_module():
    module_path = Path(__file__).resolve().parents[1] / "triton" / "pi0_spec_infer.py"
    spec = importlib.util.spec_from_file_location("pi0_spec_infer_verify_reference_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pi0_full_runtime_keeps_exact_verify_weights() -> None:
    source = (Path(__file__).resolve().parents[1] / "triton" / "pi0_infer.py").read_text(encoding="utf-8")
    for key in (
        "decoder_action_in_proj_w",
        "decoder_action_in_proj_b",
        "decoder_action_time_mlp_in_w",
        "decoder_action_time_mlp_in_b",
    ):
        assert f'"{key}"' in source


def test_spec_verify_does_not_silently_identity_fallback() -> None:
    source = (Path(__file__).resolve().parents[1] / "triton" / "pi0_spec_infer.py").read_text(encoding="utf-8")
    assert "refusing identity verify fallback" in source
    assert "_expand_batch(x0_draft.to(device=x_t_bk.device" not in source


def test_spec_verify_exact_fp32_path_is_debug_opt_in() -> None:
    source = (Path(__file__).resolve().parents[1] / "triton" / "pi0_spec_infer.py").read_text(encoding="utf-8")
    body = source.split("def _verify_exact_fp32_supported", 1)[1].split(
        "def _build_verify_action_hidden_exact",
        1,
    )[0]

    assert '_spec_triton_env("VERIFY_EXACT_FP32", "0")' in body
    assert body.find('_spec_triton_env("VERIFY_EXACT_FP32", "0")') < body.find("signal_keys")


def test_spec_verify_exact_input_path_is_debug_opt_in() -> None:
    source = (Path(__file__).resolve().parents[1] / "triton" / "pi0_spec_infer.py").read_text(encoding="utf-8")
    body = source.split("def _build_verify_action_hidden(\n", 1)[1].split(
        "def _verify_fast_path_supported",
        1,
    )[0]

    assert '_spec_triton_env("VERIFY_EXACT_INPUT", "0")' in body
    assert body.find('_spec_triton_env("VERIFY_EXACT_INPUT", "0")') < body.find("_build_verify_action_hidden_exact")


def test_triton_input_preparer_fast_matches_slow(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return

    rng = np.random.default_rng(17)
    transformed = {
        "image": {
            "base_0_rgb": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": False,
        },
        "state": rng.normal(size=(8,)).astype(np.float32),
    }
    noise = rng.normal(size=(50, 32)).astype(np.float32)
    device = torch.device("cuda:0")

    monkeypatch.setenv("SPEC_TRITON_INPUT_PREPARE_FAST", "0")
    slow_images, slow_state, slow_noise = triton_runtime.prepare_triton_inputs_from_transformed(
        transformed=transformed,
        device=device,
        action_horizon=50,
        action_dim=32,
        noise=noise,
    )
    monkeypatch.setenv("SPEC_TRITON_INPUT_PREPARE_FAST", "1")
    fast_images, fast_state, fast_noise = triton_runtime.prepare_triton_inputs_from_transformed(
        transformed=transformed,
        device=device,
        action_horizon=50,
        action_dim=32,
        noise=noise,
    )
    torch.cuda.synchronize(device)

    assert fast_images.dtype == torch.bfloat16
    assert tuple(fast_images.shape) == (2, 224, 224, 3)
    assert torch.allclose(fast_images.to(dtype=torch.float32), slow_images.to(dtype=torch.float32), atol=1e-3, rtol=0.0)
    assert torch.equal(fast_state, slow_state)
    assert torch.equal(fast_noise, slow_noise)


def _build_verify_exact_fp32_case() -> SimpleNamespace:
    module = _load_spec_verify_runtime_module()
    Pi0SpecInference = module.Pi0SpecInference
    FullCacheSnapshot = module.FullCacheSnapshot

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    checkpoint = {"language_embeds": torch.zeros((2, 2048), dtype=torch.bfloat16)}
    draft = {
        "meta": {
            "img_dim": 2048,
            "chunk_m": 50,
            "out_dim": 7,
            "draft_num_heads": 8,
            "draft_num_kv_heads": 1,
            "draft_head_dim": 256,
        },
        "draft_state_in_proj_w": torch.zeros((2048, 32), dtype=torch.float32),
        "draft_state_in_proj_b": torch.zeros((2048,), dtype=torch.float32),
        "draft_action_queries": torch.zeros((50, 2048), dtype=torch.float32),
        "draft_qkv_w": torch.zeros((2560, 2048), dtype=torch.float32),
        "draft_attn_o_w": torch.zeros((2048, 2048), dtype=torch.float32),
        "draft_ffn_gate_w": torch.zeros((16384, 2048), dtype=torch.float32),
        "draft_ffn_up_w": torch.zeros((16384, 2048), dtype=torch.float32),
        "draft_ffn_down_w": torch.zeros((2048, 16384), dtype=torch.float32),
        "draft_input_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
        "draft_post_attention_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
        "draft_action_head_w": torch.zeros((7, 2048), dtype=torch.float32),
        "draft_action_head_b": torch.zeros((7,), dtype=torch.float32),
    }
    runtime = Pi0SpecInference(checkpoint=checkpoint, draft_checkpoint=draft, num_views=2, chunk_size=50)

    hidden = 1024
    layer_count = 1
    prefix_len = 514
    action_horizon = 50
    action_dim = 32
    head_dim = 256
    ffn_hidden = 4096
    weights = {
        "decoder_state_in_proj_w": torch.randn((32, hidden), device=device, dtype=torch.float32) * 0.05,
        "decoder_state_in_proj_b": torch.randn((hidden,), device=device, dtype=torch.float32) * 0.01,
        "decoder_action_in_proj_w": torch.randn((32, hidden), device=device, dtype=torch.float32) * 0.05,
        "decoder_action_in_proj_b": torch.randn((hidden,), device=device, dtype=torch.float32) * 0.01,
        "decoder_action_time_mlp_in_w": torch.randn((hidden * 2, hidden), device=device, dtype=torch.float32) * 0.05,
        "decoder_action_time_mlp_in_b": torch.randn((hidden,), device=device, dtype=torch.float32) * 0.01,
        "decoder_action_fused_in_proj_w": torch.zeros((32, hidden), device=device, dtype=torch.float32),
        "decoder_action_fused_time_biases": torch.zeros((10, hidden), device=device, dtype=torch.float32),
        "decoder_action_mlp_w": torch.randn((hidden, hidden), device=device, dtype=torch.float32) * 0.05,
        "decoder_action_mlp_b": torch.randn((hidden,), device=device, dtype=torch.float32) * 0.01,
        "decoder_attn_qkv_w": torch.zeros((layer_count, hidden, 2560), device=device, dtype=torch.float32),
        "decoder_attn_o_w": torch.zeros((layer_count, hidden * 2, hidden), device=device, dtype=torch.float32),
        "decoder_ffn_gate_w": torch.zeros((layer_count, hidden, ffn_hidden), device=device, dtype=torch.float32),
        "decoder_ffn_up_w": torch.zeros((layer_count, hidden, ffn_hidden), device=device, dtype=torch.float32),
        "decoder_ffn_down_w": torch.zeros((layer_count, ffn_hidden, hidden), device=device, dtype=torch.float32),
        "decoder_action_fused_out_proj_w": torch.randn((hidden, action_dim), device=device, dtype=torch.float32) * 0.05,
        "decoder_action_fused_out_proj_b": torch.randn((action_dim,), device=device, dtype=torch.float32) * 0.01,
    }

    class FullRuntime:
        pass

    full_runtime = FullRuntime()
    full_runtime.weights = weights
    cache = FullCacheSnapshot(
        encoder_seq_len=prefix_len,
        encoder_x=None,
        encoder_k=torch.zeros((layer_count, prefix_len, head_dim), device=device, dtype=torch.float32),
        encoder_v=torch.zeros((layer_count, prefix_len, head_dim), device=device, dtype=torch.float32),
    )
    noise = torch.randn((1, action_horizon, action_dim), device=device, dtype=torch.float32)
    x0_draft = torch.randn((1, action_horizon, action_dim), device=device, dtype=torch.float32)
    state = torch.randn((32,), device=device, dtype=torch.float32)
    t_list = (0.10, 0.05)

    return SimpleNamespace(
        device=device,
        runtime=runtime,
        full_runtime=full_runtime,
        cache=cache,
        noise=noise,
        x0_draft=x0_draft,
        state=state,
        t_list=t_list,
        weights=weights,
        hidden=hidden,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )


def test_spec_runtime_draft_graph_matches_plain_draft(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return

    module = _load_spec_verify_runtime_module()
    Pi0SpecInference = module.Pi0SpecInference
    PrefixContext = module.PrefixContext
    device = torch.device("cuda:0")
    torch.manual_seed(17)

    hidden = 8
    draft = {
        "meta": {
            "img_dim": hidden,
            "chunk_m": 4,
            "out_dim": 7,
            "draft_num_heads": 2,
            "draft_num_kv_heads": 1,
            "draft_head_dim": 4,
        },
        "draft_state_in_proj_w": torch.randn((hidden, 32), dtype=torch.float32) * 0.03,
        "draft_state_in_proj_b": torch.randn((hidden,), dtype=torch.float32) * 0.01,
        "draft_action_queries": torch.randn((4, hidden), dtype=torch.float32) * 0.03,
        "draft_qkv_w": torch.randn((16, hidden), dtype=torch.float32) * 0.03,
        "draft_attn_o_w": torch.randn((hidden, hidden), dtype=torch.float32) * 0.03,
        "draft_ffn_gate_w": torch.randn((16, hidden), dtype=torch.float32) * 0.03,
        "draft_ffn_up_w": torch.randn((16, hidden), dtype=torch.float32) * 0.03,
        "draft_ffn_down_w": torch.randn((hidden, 16), dtype=torch.float32) * 0.03,
        "draft_input_layernorm_w": torch.ones((hidden,), dtype=torch.bfloat16),
        "draft_post_attention_layernorm_w": torch.ones((hidden,), dtype=torch.bfloat16),
        "draft_action_head_w": torch.randn((7, hidden), dtype=torch.float32) * 0.03,
        "draft_action_head_b": torch.randn((7,), dtype=torch.float32) * 0.01,
    }
    runtime = Pi0SpecInference(
        checkpoint={"language_embeds": torch.zeros((2, hidden), dtype=torch.bfloat16)},
        draft_checkpoint=draft,
        num_views=2,
        chunk_size=50,
    )
    prefix = PrefixContext(
        prefix_embs=torch.randn((1, 6, hidden), device=device, dtype=torch.float32) * 0.03,
        prefix_pad_masks=torch.ones((1, 6), device=device, dtype=torch.bool),
        prefix_att_masks=torch.zeros((1, 6), device=device, dtype=torch.bool),
    )
    state = torch.randn((32,), device=device, dtype=torch.float32) * 0.03

    monkeypatch.setenv("SPEC_TRITON_DRAFT_GRAPH", "0")
    plain = runtime._run_draft_block_maybe_graph(prefix=prefix, observation_state_normalized=state)
    monkeypatch.setenv("SPEC_TRITON_DRAFT_GRAPH", "1")
    graph = runtime._run_draft_block_maybe_graph(prefix=prefix, observation_state_normalized=state)

    assert runtime._draft_graph is not None
    assert not runtime._draft_graph_failed
    assert torch.equal(graph, plain)


def _compute_verify_reference_outputs(case: SimpleNamespace) -> SimpleNamespace:
    from openpi.models_pytorch.spec_pi0_pytorch import _compute_radius_prefix_acceptance
    from openpi.models_pytorch.spec_pi0_pytorch import _make_speculative_metrics
    from openpi.models_pytorch.spec_pi0_pytorch import _stitch_radius_prefix_output

    runtime = case.runtime
    weights = case.weights
    device = case.device
    batch_k = len(case.t_list)

    x_t_bk, timestep_bk = runtime._build_verify_batch(
        noise=case.noise,
        x0_draft=case.x0_draft,
        t_list=case.t_list,
    )
    state_bk = case.state.unsqueeze(0).expand(batch_k, -1).contiguous()
    action_rows = x_t_bk.reshape(batch_k * case.action_horizon, case.action_dim)
    action_emb = action_rows @ weights["decoder_action_in_proj_w"] + weights["decoder_action_in_proj_b"]
    time_emb = runtime._sinusoidal_time_embedding(
        timestep=timestep_bk.to(device=device, dtype=torch.float32),
        hidden_size=case.hidden,
    )
    mlp_in_w = weights["decoder_action_time_mlp_in_w"]
    action_w = mlp_in_w[: case.hidden]
    time_w = mlp_in_w[case.hidden :]
    preact = (
        action_emb @ action_w
        + time_emb.repeat_interleave(case.action_horizon, dim=0) @ time_w
        + weights["decoder_action_time_mlp_in_b"]
    )
    action_hidden = torch.nn.functional.silu(preact) @ weights["decoder_action_mlp_w"] + weights["decoder_action_mlp_b"]
    state_hidden = state_bk @ weights["decoder_state_in_proj_w"] + weights["decoder_state_in_proj_b"]
    x_ref = torch.empty((batch_k, case.action_horizon + 1, case.hidden), device=device, dtype=torch.float32)
    x_ref[:, 0, :] = state_hidden
    x_ref[:, 1:, :] = action_hidden.view(batch_k, case.action_horizon, case.hidden)
    x_ref_actions = x_ref[:, 1:, :]
    rms = torch.rsqrt(x_ref_actions.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    delta = x_ref_actions * rms
    delta = delta @ weights["decoder_action_fused_out_proj_w"] + weights["decoder_action_fused_out_proj_b"]
    velocity = delta * float(runtime._triton_delta_to_velocity_scale(weights=weights))
    x0_hat = x_t_bk - timestep_bk[:, None, None] * velocity
    x0_hat = x0_hat.view(1, batch_k, case.action_horizon, case.action_dim)

    accepted_prefix_len, dist = _compute_radius_prefix_acceptance(
        x0_draft=case.x0_draft,
        x0_hat=x0_hat,
        tau_radius=0.3,
        dist_dims=7,
        eval_h=12,
    )
    actions = _stitch_radius_prefix_output(
        x0_draft=case.x0_draft,
        x0_tail=x0_hat.mean(dim=1),
        accepted_prefix_len=accepted_prefix_len,
    )
    metrics = _make_speculative_metrics(
        radius_dist=dist.mean(),
        accepted_prefix_len_mean=accepted_prefix_len.to(dtype=torch.float32).mean(),
        gripper_switch_cut_rate=torch.zeros((), device=device, dtype=torch.float32),
        scheduled_full_fallback_gripper=torch.zeros((), device=device, dtype=torch.float32),
        gripper_verify_stop_rate=torch.zeros((), device=device, dtype=torch.float32),
    )
    return SimpleNamespace(
        x_t_bk=x_t_bk,
        timestep_bk=timestep_bk,
        x0_hat=x0_hat,
        accepted_prefix_len=accepted_prefix_len,
        actions=actions,
        metrics=metrics,
    )


def test_spec_runtime_fused_postprocess_matches_current_semantics(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return

    module = _load_spec_verify_runtime_module()
    runtime = module.Pi0SpecInference(
        checkpoint={"language_embeds": torch.zeros((2, 2048), dtype=torch.bfloat16)},
        draft_checkpoint={
            "meta": {
                "img_dim": 2048,
                "chunk_m": 50,
                "out_dim": 7,
                "draft_num_heads": 8,
                "draft_num_kv_heads": 1,
                "draft_head_dim": 256,
            },
            "draft_state_in_proj_w": torch.zeros((2048, 32), dtype=torch.float32),
            "draft_state_in_proj_b": torch.zeros((2048,), dtype=torch.float32),
            "draft_action_queries": torch.zeros((50, 2048), dtype=torch.float32),
            "draft_qkv_w": torch.zeros((2560, 2048), dtype=torch.float32),
            "draft_attn_o_w": torch.zeros((2048, 2048), dtype=torch.float32),
            "draft_ffn_gate_w": torch.zeros((16384, 2048), dtype=torch.float32),
            "draft_ffn_up_w": torch.zeros((16384, 2048), dtype=torch.float32),
            "draft_ffn_down_w": torch.zeros((2048, 16384), dtype=torch.float32),
            "draft_input_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
            "draft_post_attention_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
            "draft_action_head_w": torch.zeros((7, 2048), dtype=torch.float32),
            "draft_action_head_b": torch.zeros((7,), dtype=torch.float32),
        },
        num_views=2,
        chunk_size=50,
    )
    device = torch.device("cuda:0")
    torch.manual_seed(123)
    x0_draft = torch.randn((1, 50, 32), device=device, dtype=torch.float32) * 0.02
    x0_hat = x0_draft[:, None, :, :].expand(-1, 2, -1, -1).contiguous()
    x0_hat = x0_hat + torch.randn_like(x0_hat) * 0.01
    x0_hat[:, 0, 2, 6] = 1.0
    last_gripper = torch.tensor([-1.0], device=device, dtype=torch.float32)

    kwargs = {
        "x0_hat": x0_hat,
        "x0_draft": x0_draft,
        "tau_radius": 0.3,
        "dist_dims": 7,
        "max_exec_steps": 12,
        "last_gripper": last_gripper,
        "gripper_switch_threshold": 0.0,
        "enable_gripper_verify": True,
        "enable_gripper_post_verify": True,
    }
    monkeypatch.setenv("SPEC_TRITON_POSTPROCESS_FUSED", "0")
    reference = runtime._run_verify_postprocess_fused(**kwargs)
    assert reference is None

    from openpi.models_pytorch.spec_pi0_pytorch import _compute_radius_prefix_acceptance
    from openpi.models_pytorch.spec_pi0_pytorch import _detect_verify_gripper_switch_any_k
    from openpi.models_pytorch.spec_pi0_pytorch import _make_speculative_metrics
    from openpi.models_pytorch.spec_pi0_pytorch import _stitch_radius_prefix_output
    from openpi.models_pytorch.spec_pi0_pytorch import _truncate_accepted_prefix_on_gripper_switch

    accepted_prefix_len, dist = _compute_radius_prefix_acceptance(
        x0_draft=x0_draft,
        x0_hat=x0_hat,
        tau_radius=0.3,
        dist_dims=7,
        eval_h=12,
    )
    x0_tail = x0_hat.mean(dim=1)
    actions = _stitch_radius_prefix_output(
        x0_draft=x0_draft,
        x0_tail=x0_tail,
        accepted_prefix_len=accepted_prefix_len,
    )
    gripper_verify_stop_mask = _detect_verify_gripper_switch_any_k(
        x0_hat=x0_hat,
        gripper_prev=last_gripper,
        gripper_switch_threshold=0.0,
        eval_h=12,
    )
    accepted_prefix_len = torch.where(
        gripper_verify_stop_mask,
        torch.zeros_like(accepted_prefix_len),
        accepted_prefix_len,
    )
    actions = torch.where(gripper_verify_stop_mask[:, None, None], x0_tail, actions)
    accepted_after_cut, gripper_switch_cut_mask = _truncate_accepted_prefix_on_gripper_switch(
        x0_out=actions,
        accepted_prefix_len=accepted_prefix_len,
        gripper_prev=last_gripper,
        gripper_switch_threshold=0.0,
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

    monkeypatch.setenv("SPEC_TRITON_POSTPROCESS_FUSED", "1")
    fused = runtime._run_verify_postprocess_fused(**kwargs)
    assert fused is not None
    assert torch.equal(fused.accepted_prefix_len, accepted_prefix_len)
    assert torch.allclose(fused.actions, actions)
    assert torch.allclose(fused.metrics, metrics)


def test_spec_runtime_verify_fast_bf16_matches_generic_with_nonzero_decoder(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return

    monkeypatch.setenv("SPEC_TRITON_VERIFY_GRAPH", "0")
    monkeypatch.setenv("SPEC_TRITON_POSTPROCESS_FUSED", "0")
    module = _load_spec_verify_runtime_module()
    runtime = module.Pi0SpecInference(
        checkpoint={"language_embeds": torch.zeros((2, 2048), dtype=torch.bfloat16)},
        draft_checkpoint={
            "meta": {
                "img_dim": 2048,
                "chunk_m": 50,
                "out_dim": 7,
                "draft_num_heads": 8,
                "draft_num_kv_heads": 1,
                "draft_head_dim": 256,
            },
            "draft_state_in_proj_w": torch.zeros((2048, 32), dtype=torch.float32),
            "draft_state_in_proj_b": torch.zeros((2048,), dtype=torch.float32),
            "draft_action_queries": torch.zeros((50, 2048), dtype=torch.float32),
            "draft_qkv_w": torch.zeros((2560, 2048), dtype=torch.float32),
            "draft_attn_o_w": torch.zeros((2048, 2048), dtype=torch.float32),
            "draft_ffn_gate_w": torch.zeros((16384, 2048), dtype=torch.float32),
            "draft_ffn_up_w": torch.zeros((16384, 2048), dtype=torch.float32),
            "draft_ffn_down_w": torch.zeros((2048, 16384), dtype=torch.float32),
            "draft_input_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
            "draft_post_attention_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
            "draft_action_head_w": torch.zeros((7, 2048), dtype=torch.float32),
            "draft_action_head_b": torch.zeros((7,), dtype=torch.float32),
        },
        num_views=2,
        chunk_size=50,
    )
    device = torch.device("cuda:0")
    torch.manual_seed(7)
    layer_count = 1
    shapes = {
        "decoder_state_in_proj_w": (32, 1024),
        "decoder_state_in_proj_b": (1024,),
        "decoder_action_in_proj_w": (32, 1024),
        "decoder_action_in_proj_b": (1024,),
        "decoder_action_time_mlp_in_w": (2048, 1024),
        "decoder_action_time_mlp_in_b": (1024,),
        "decoder_action_fused_in_proj_w": (32, 1024),
        "decoder_action_fused_time_biases": (10, 1024),
        "decoder_action_mlp_w": (1024, 1024),
        "decoder_action_mlp_b": (1024,),
        "decoder_attn_qkv_w": (layer_count, 1024, 2560),
        "decoder_attn_o_w": (layer_count, 2048, 1024),
        "decoder_ffn_gate_w": (layer_count, 1024, 4096),
        "decoder_ffn_up_w": (layer_count, 1024, 4096),
        "decoder_ffn_down_w": (layer_count, 4096, 1024),
        "decoder_action_fused_out_proj_w": (1024, 32),
        "decoder_action_fused_out_proj_b": (32,),
    }
    weights = {
        key: (torch.randn(shape, device=device, dtype=torch.float32) * 0.001).to(torch.bfloat16).contiguous()
        for key, shape in shapes.items()
    }

    class FullRuntime:
        pass

    full_runtime = FullRuntime()
    full_runtime.weights = weights
    cache = module.FullCacheSnapshot(
        encoder_seq_len=514,
        encoder_x=None,
        encoder_k=(torch.randn((layer_count, 514, 256), device=device, dtype=torch.float32) * 0.001)
        .to(torch.bfloat16)
        .contiguous(),
        encoder_v=(torch.randn((layer_count, 514, 256), device=device, dtype=torch.float32) * 0.001)
        .to(torch.bfloat16)
        .contiguous(),
    )
    state = torch.randn((32,), device=device, dtype=torch.float32) * 0.001
    noise = torch.randn((1, 50, 32), device=device, dtype=torch.float32) * 0.01
    x0_draft = torch.randn((1, 50, 32), device=device, dtype=torch.float32) * 0.01
    images = torch.empty((2, 224, 224, 3), device=device, dtype=torch.float32)

    fast = runtime.run_verify(cache, images, state, noise, x0_draft, (0.10, 0.05), full_runtime=full_runtime)
    original = runtime._verify_fast_path_supported
    runtime._verify_fast_path_supported = staticmethod(lambda **_kwargs: False)
    try:
        generic = runtime.run_verify(cache, images, state, noise, x0_draft, (0.10, 0.05), full_runtime=full_runtime)
    finally:
        runtime._verify_fast_path_supported = original

    assert torch.allclose(fast, generic, atol=3e-4, rtol=0.0)


def test_spec_runtime_session_runs_draft_with_timing(tmp_path: Path) -> None:
    session = _build_spec_session(tmp_path)
    prepared = session.prepare_observation(
        images=torch.zeros((2, 4, 4, 3), dtype=torch.float32),
        state=torch.zeros((32,), dtype=torch.float32),
    )

    x0_draft, timing = session.run_draft_with_timing(
        prepared=prepared,
    )

    assert x0_draft.shape == (1, 50, 32)
    assert timing["encoder_ms"] >= 0.0
    assert timing["draft_ms"] >= 0.0


def test_spec_runtime_session_runs_draft_without_pytorch_draft_head(monkeypatch, tmp_path: Path) -> None:
    from openpi.models_pytorch import draft as draft_module

    def fail_forward(self, *args, **kwargs):
        raise AssertionError("runtime draft path should not call DraftChunkHead.forward")

    monkeypatch.setattr(draft_module.DraftChunkHead, "forward", fail_forward, raising=True)

    session = _build_spec_session(tmp_path)
    prepared = session.prepare_observation(
        images=torch.zeros((2, 4, 4, 3), dtype=torch.float32),
        state=torch.zeros((32,), dtype=torch.float32),
    )

    x0_draft, timing = session.run_draft_with_timing(
        prepared=prepared,
    )

    assert x0_draft.shape == (1, 50, 32)
    assert timing["draft_ms"] >= 0.0


def test_spec_runtime_draft_path_does_not_use_torch_attention_ops(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("draft runtime should use Triton attention kernels")

    monkeypatch.setattr(torch, "einsum", fail, raising=True)
    monkeypatch.setattr(torch, "softmax", fail, raising=True)

    session = _build_spec_session(tmp_path)
    prepared = session.prepare_observation(
        images=torch.zeros((2, 4, 4, 3), dtype=torch.float32),
        state=torch.zeros((32,), dtype=torch.float32),
    )

    x0_draft, timing = session.run_draft_with_timing(
        prepared=prepared,
    )

    assert x0_draft.shape == (1, 50, 32)
    assert timing["draft_ms"] >= 0.0


def test_spec_runtime_verify_refuses_missing_full_cache_snapshot(tmp_path: Path) -> None:
    session = _build_spec_session(tmp_path)
    prepared = session.prepare_observation(
        images=torch.zeros((2, 4, 4, 3), dtype=torch.float32),
        state=torch.zeros((32,), dtype=torch.float32),
    )

    full_snapshot = session.capture_full_cache_snapshot()
    try:
        session.run_verify_with_timing(
            cache_snapshot=full_snapshot,
            prepared=prepared,
            noise=torch.zeros((1, 50, 32), dtype=torch.float32),
            x0_draft=torch.zeros((1, 50, 32), dtype=torch.float32),
            t_list=(0.10, 0.05),
        )
    except RuntimeError as exc:
        assert "refusing identity verify fallback" in str(exc)
        assert "cache_snapshot.encoder_k is missing" in str(exc)
    else:
        raise AssertionError("verify should refuse a missing full cache snapshot")


def test_spec_runtime_verify_refuses_missing_full_weights_before_torch_ops(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("verify runtime should use Triton kernels")

    monkeypatch.setattr(torch.nn.functional, "silu", fail, raising=True)
    monkeypatch.setattr(torch, "softmax", fail, raising=True)

    session = _build_spec_session(tmp_path)
    prepared = session.prepare_observation(
        images=torch.zeros((2, 4, 4, 3), dtype=torch.float32),
        state=torch.zeros((32,), dtype=torch.float32),
    )

    full_snapshot = session.capture_full_cache_snapshot()
    try:
        session.run_verify_with_timing(
            cache_snapshot=full_snapshot,
            prepared=prepared,
            noise=torch.zeros((1, 50, 32), dtype=torch.float32),
            x0_draft=torch.zeros((1, 50, 32), dtype=torch.float32),
            t_list=(0.10, 0.05),
        )
    except RuntimeError as exc:
        assert "refusing identity verify fallback" in str(exc)
        assert "missing decoder weights" in str(exc)
    else:
        raise AssertionError("verify should refuse missing full decoder weights")


def test_spec_runtime_verify_kernel_path_does_not_use_torch_pointwise_or_softmax(
    monkeypatch, tmp_path: Path
) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("verify runtime should use Triton kernels")

    monkeypatch.setattr(torch.nn.functional, "silu", fail, raising=True)
    monkeypatch.setattr(torch, "softmax", fail, raising=True)

    session = _build_spec_session(tmp_path)
    runtime = session.draft_runtime
    cache_snapshot = runtime.capture_full_cache_snapshot(None)
    cache_snapshot = type(cache_snapshot)(
        encoder_seq_len=3,
        encoder_x=None,
        encoder_k=torch.zeros((1, 3, 4), dtype=torch.float32, device=runtime.device),
        encoder_v=torch.zeros((1, 3, 4), dtype=torch.float32, device=runtime.device),
    )

    class FakeFullRuntime:
        weights = {
            "decoder_state_in_proj_w": torch.zeros((32, 8), dtype=torch.float32, device=runtime.device),
            "decoder_state_in_proj_b": torch.zeros((8,), dtype=torch.float32, device=runtime.device),
            "decoder_action_in_proj_w": torch.zeros((32, 8), dtype=torch.float32, device=runtime.device),
            "decoder_action_in_proj_b": torch.zeros((8,), dtype=torch.float32, device=runtime.device),
            "decoder_action_time_mlp_in_w": torch.zeros((16, 8), dtype=torch.float32, device=runtime.device),
            "decoder_action_time_mlp_in_b": torch.zeros((8,), dtype=torch.float32, device=runtime.device),
            "decoder_action_fused_in_proj_w": torch.zeros((32, 8), dtype=torch.float32, device=runtime.device),
            "decoder_action_fused_time_biases": torch.zeros((10, 8), dtype=torch.float32, device=runtime.device),
            "decoder_action_mlp_w": torch.zeros((8, 8), dtype=torch.float32, device=runtime.device),
            "decoder_action_mlp_b": torch.zeros((8,), dtype=torch.float32, device=runtime.device),
            "decoder_attn_qkv_w": torch.zeros((1, 8, 16), dtype=torch.float32, device=runtime.device),
            "decoder_attn_o_w": torch.zeros((1, 8, 8), dtype=torch.float32, device=runtime.device),
            "decoder_ffn_gate_w": torch.zeros((1, 8, 16), dtype=torch.float32, device=runtime.device),
            "decoder_ffn_up_w": torch.zeros((1, 8, 16), dtype=torch.float32, device=runtime.device),
            "decoder_ffn_down_w": torch.zeros((1, 16, 8), dtype=torch.float32, device=runtime.device),
            "decoder_action_fused_out_proj_w": torch.zeros((8, 32), dtype=torch.float32, device=runtime.device),
            "decoder_action_fused_out_proj_b": torch.zeros((32,), dtype=torch.float32, device=runtime.device),
        }

    x0_hat, timing = runtime.run_verify_with_timing(
        cache_snapshot,
        torch.zeros((2, 4, 4, 3), dtype=torch.float32),
        torch.zeros((32,), dtype=torch.float32),
        torch.zeros((1, 50, 32), dtype=torch.float32),
        torch.zeros((1, 50, 32), dtype=torch.float32),
        (0.10, 0.05),
        full_runtime=FakeFullRuntime(),
    )

    assert x0_hat.shape == (1, 2, 50, 32)
    assert timing["action_verify_ms"] >= 0.0


def test_spec_runtime_verify_fast_kernel_flow_runs_without_illegal_memory() -> None:
    script = r"""
import importlib.util
from pathlib import Path
import torch

module_path = Path("scripts/spec/triton/pi0_spec_infer.py").resolve()
spec = importlib.util.spec_from_file_location('pi0_spec_infer_dbg', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
orig_rms_qkv_rope_kernel = module._rms_qkv_rope_kernel
def force_fast(*args, **kwargs):
    kwargs['safe_kernel'] = False
    return orig_rms_qkv_rope_kernel(*args, **kwargs)
module._rms_qkv_rope_kernel = force_fast
Pi0SpecInference = module.Pi0SpecInference
FullCacheSnapshot = module.FullCacheSnapshot
PrefixContext = module.PrefixContext

device = torch.device('cuda:0')
checkpoint = {'language_embeds': torch.zeros((2, 2048), dtype=torch.bfloat16)}
draft = {
    'meta': {'img_dim': 2048, 'chunk_m': 50, 'out_dim': 7, 'draft_num_heads': 8, 'draft_num_kv_heads': 1, 'draft_head_dim': 256},
    'draft_state_in_proj_w': torch.zeros((2048, 32), dtype=torch.float32),
    'draft_state_in_proj_b': torch.zeros((2048,), dtype=torch.float32),
    'draft_action_queries': torch.zeros((50, 2048), dtype=torch.float32),
    'draft_qkv_w': torch.zeros((2560, 2048), dtype=torch.float32),
    'draft_attn_o_w': torch.zeros((2048, 2048), dtype=torch.float32),
    'draft_ffn_gate_w': torch.zeros((16384, 2048), dtype=torch.float32),
    'draft_ffn_up_w': torch.zeros((16384, 2048), dtype=torch.float32),
    'draft_ffn_down_w': torch.zeros((2048, 16384), dtype=torch.float32),
    'draft_input_layernorm_w': torch.ones((2048,), dtype=torch.bfloat16),
    'draft_post_attention_layernorm_w': torch.ones((2048,), dtype=torch.bfloat16),
    'draft_action_head_w': torch.zeros((7, 2048), dtype=torch.float32),
    'draft_action_head_b': torch.zeros((7,), dtype=torch.float32),
}
runtime = Pi0SpecInference(checkpoint=checkpoint, draft_checkpoint=draft, num_views=2, chunk_size=50)
prefix = PrefixContext(
    prefix_embs=torch.zeros((1, 514, 2048), device=device, dtype=torch.float32),
    prefix_pad_masks=torch.ones((1, 514), device=device, dtype=torch.bool),
    prefix_att_masks=torch.zeros((1, 514), device=device, dtype=torch.bool),
)
state = torch.zeros((32,), device=device, dtype=torch.float32)
x0 = runtime._run_draft_block(
    prefix=prefix,
    observation_state_normalized=state,
)
layer_count = 16
weights = {
    'decoder_state_in_proj_w': torch.zeros((32, 1024), device=device, dtype=torch.float32),
    'decoder_state_in_proj_b': torch.zeros((1024,), device=device, dtype=torch.float32),
    'decoder_action_in_proj_w': torch.zeros((32, 1024), device=device, dtype=torch.float32),
    'decoder_action_in_proj_b': torch.zeros((1024,), device=device, dtype=torch.float32),
    'decoder_action_time_mlp_in_w': torch.zeros((2048, 1024), device=device, dtype=torch.float32),
    'decoder_action_time_mlp_in_b': torch.zeros((1024,), device=device, dtype=torch.float32),
    'decoder_action_fused_in_proj_w': torch.zeros((32, 1024), device=device, dtype=torch.float32),
    'decoder_action_fused_time_biases': torch.zeros((10, 1024), device=device, dtype=torch.float32),
    'decoder_action_mlp_w': torch.zeros((1024, 1024), device=device, dtype=torch.float32),
    'decoder_action_mlp_b': torch.zeros((1024,), device=device, dtype=torch.float32),
    'decoder_attn_qkv_w': torch.zeros((layer_count, 1024, 2560), device=device, dtype=torch.float32),
    'decoder_attn_o_w': torch.zeros((layer_count, 2048, 1024), device=device, dtype=torch.float32),
    'decoder_ffn_gate_w': torch.zeros((layer_count, 1024, 4096), device=device, dtype=torch.float32),
    'decoder_ffn_up_w': torch.zeros((layer_count, 1024, 4096), device=device, dtype=torch.float32),
    'decoder_ffn_down_w': torch.zeros((layer_count, 4096, 1024), device=device, dtype=torch.float32),
    'decoder_action_fused_out_proj_w': torch.zeros((1024, 32), device=device, dtype=torch.float32),
    'decoder_action_fused_out_proj_b': torch.zeros((32,), device=device, dtype=torch.float32),
}
class FullRuntime:
    pass
full_runtime = FullRuntime()
full_runtime.weights = weights
cache = FullCacheSnapshot(
    encoder_seq_len=514,
    encoder_x=None,
    encoder_k=torch.zeros((layer_count, 514, 256), device=device, dtype=torch.float32),
    encoder_v=torch.zeros((layer_count, 514, 256), device=device, dtype=torch.float32),
)
result = runtime.run_verify_semantics(
    cache_snapshot=cache,
    observation_images_normalized=torch.zeros((2, 224, 224, 3), device=device, dtype=torch.float32),
    observation_state_normalized=state,
    noise=torch.zeros((1, 50, 32), device=device, dtype=torch.float32),
    x0_draft=x0,
    t_list=(0.10, 0.05),
    tau_radius=0.3,
    dist_dims=7,
    max_exec_steps=12,
    last_gripper=None,
    gripper_switch_threshold=0.0,
    enable_gripper_verify=True,
    enable_gripper_post_verify=True,
    full_runtime=full_runtime,
)
torch.cuda.synchronize(device)
print(result.actions.shape)
"""
    env = dict(os.environ)
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "torch.Size([1, 50, 32])" in completed.stdout


def test_spec_runtime_verify_fast_path_avoids_generic_attention(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return

    module_path = Path(__file__).resolve().parents[1] / "triton" / "pi0_spec_infer.py"
    spec = importlib.util.spec_from_file_location("pi0_spec_infer_fast_verify_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    Pi0SpecInference = module.Pi0SpecInference
    FullCacheSnapshot = module.FullCacheSnapshot
    PrefixContext = module.PrefixContext

    device = torch.device("cuda:0")
    checkpoint = {"language_embeds": torch.zeros((2, 2048), dtype=torch.bfloat16)}
    draft = {
        "meta": {
            "img_dim": 2048,
            "chunk_m": 50,
            "out_dim": 7,
            "draft_num_heads": 8,
            "draft_num_kv_heads": 1,
            "draft_head_dim": 256,
        },
        "draft_state_in_proj_w": torch.zeros((2048, 32), dtype=torch.float32),
        "draft_state_in_proj_b": torch.zeros((2048,), dtype=torch.float32),
        "draft_action_queries": torch.zeros((50, 2048), dtype=torch.float32),
        "draft_qkv_w": torch.zeros((2560, 2048), dtype=torch.float32),
        "draft_attn_o_w": torch.zeros((2048, 2048), dtype=torch.float32),
        "draft_ffn_gate_w": torch.zeros((16384, 2048), dtype=torch.float32),
        "draft_ffn_up_w": torch.zeros((16384, 2048), dtype=torch.float32),
        "draft_ffn_down_w": torch.zeros((2048, 16384), dtype=torch.float32),
        "draft_input_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
        "draft_post_attention_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
        "draft_action_head_w": torch.zeros((7, 2048), dtype=torch.float32),
        "draft_action_head_b": torch.zeros((7,), dtype=torch.float32),
    }
    runtime = Pi0SpecInference(checkpoint=checkpoint, draft_checkpoint=draft, num_views=2, chunk_size=50)

    prefix = PrefixContext(
        prefix_embs=torch.zeros((1, 514, 2048), device=device, dtype=torch.float32),
        prefix_pad_masks=torch.ones((1, 514), device=device, dtype=torch.bool),
        prefix_att_masks=torch.zeros((1, 514), device=device, dtype=torch.bool),
    )
    state = torch.zeros((32,), device=device, dtype=torch.float32)
    x0 = runtime._run_draft_block(prefix=prefix, observation_state_normalized=state)

    def fail_attention(*args, **kwargs):
        raise AssertionError("verify fast path should not call the generic attention kernel")

    monkeypatch.setattr(runtime, "_attention_kernel", fail_attention, raising=True)
    layer_count = 18
    weights = {
        "decoder_state_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_state_in_proj_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_action_in_proj_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_time_mlp_in_w": torch.zeros((2048, 1024), device=device, dtype=torch.float32),
        "decoder_action_time_mlp_in_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_fused_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_action_fused_time_biases": torch.zeros((10, 1024), device=device, dtype=torch.float32),
        "decoder_action_mlp_w": torch.zeros((1024, 1024), device=device, dtype=torch.float32),
        "decoder_action_mlp_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_attn_qkv_w": torch.zeros((layer_count, 1024, 2560), device=device, dtype=torch.float32),
        "decoder_attn_o_w": torch.zeros((layer_count, 2048, 1024), device=device, dtype=torch.float32),
        "decoder_ffn_gate_w": torch.zeros((layer_count, 1024, 4096), device=device, dtype=torch.float32),
        "decoder_ffn_up_w": torch.zeros((layer_count, 1024, 4096), device=device, dtype=torch.float32),
        "decoder_ffn_down_w": torch.zeros((layer_count, 4096, 1024), device=device, dtype=torch.float32),
        "decoder_action_fused_out_proj_w": torch.zeros((1024, 32), device=device, dtype=torch.float32),
        "decoder_action_fused_out_proj_b": torch.zeros((32,), device=device, dtype=torch.float32),
    }

    class FullRuntime:
        pass

    full_runtime = FullRuntime()
    full_runtime.weights = weights
    cache = FullCacheSnapshot(
        encoder_seq_len=514,
        encoder_x=None,
        encoder_k=torch.zeros((layer_count, 514, 256), device=device, dtype=torch.float32),
        encoder_v=torch.zeros((layer_count, 514, 256), device=device, dtype=torch.float32),
    )

    result = runtime.run_verify_semantics(
        cache_snapshot=cache,
        observation_images_normalized=torch.zeros((2, 224, 224, 3), device=device, dtype=torch.float32),
        observation_state_normalized=state,
        noise=torch.zeros((1, 50, 32), device=device, dtype=torch.float32),
        x0_draft=x0,
        t_list=(0.10, 0.05),
        tau_radius=0.3,
        dist_dims=7,
        max_exec_steps=12,
        last_gripper=None,
        gripper_switch_threshold=0.0,
        enable_gripper_verify=True,
        enable_gripper_post_verify=True,
        full_runtime=full_runtime,
    )

    assert result.actions.shape == (1, 50, 32)


def test_spec_runtime_verify_fast_path_matches_generic_with_gripper_stop() -> None:
    if not torch.cuda.is_available():
        return

    module_path = Path(__file__).resolve().parents[1] / "triton" / "pi0_spec_infer.py"
    spec = importlib.util.spec_from_file_location("pi0_spec_infer_gripper_verify_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    Pi0SpecInference = module.Pi0SpecInference
    FullCacheSnapshot = module.FullCacheSnapshot
    PrefixContext = module.PrefixContext

    device = torch.device("cuda:0")
    checkpoint = {"language_embeds": torch.zeros((2, 2048), dtype=torch.bfloat16)}
    draft = {
        "meta": {
            "img_dim": 2048,
            "chunk_m": 50,
            "out_dim": 7,
            "draft_num_heads": 8,
            "draft_num_kv_heads": 1,
            "draft_head_dim": 256,
        },
        "draft_state_in_proj_w": torch.zeros((2048, 32), dtype=torch.float32),
        "draft_state_in_proj_b": torch.zeros((2048,), dtype=torch.float32),
        "draft_action_queries": torch.zeros((50, 2048), dtype=torch.float32),
        "draft_qkv_w": torch.zeros((2560, 2048), dtype=torch.float32),
        "draft_attn_o_w": torch.zeros((2048, 2048), dtype=torch.float32),
        "draft_ffn_gate_w": torch.zeros((16384, 2048), dtype=torch.float32),
        "draft_ffn_up_w": torch.zeros((16384, 2048), dtype=torch.float32),
        "draft_ffn_down_w": torch.zeros((2048, 16384), dtype=torch.float32),
        "draft_input_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
        "draft_post_attention_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
        "draft_action_head_w": torch.zeros((7, 2048), dtype=torch.float32),
        "draft_action_head_b": torch.zeros((7,), dtype=torch.float32),
    }
    runtime = Pi0SpecInference(checkpoint=checkpoint, draft_checkpoint=draft, num_views=2, chunk_size=50)

    prefix = PrefixContext(
        prefix_embs=torch.zeros((1, 514, 2048), device=device, dtype=torch.float32),
        prefix_pad_masks=torch.ones((1, 514), device=device, dtype=torch.bool),
        prefix_att_masks=torch.zeros((1, 514), device=device, dtype=torch.bool),
    )
    state = torch.zeros((32,), device=device, dtype=torch.float32)
    x0 = runtime._run_draft_block(prefix=prefix, observation_state_normalized=state)
    layer_count = 18
    weights = {
        "decoder_state_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_state_in_proj_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_action_in_proj_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_time_mlp_in_w": torch.zeros((2048, 1024), device=device, dtype=torch.float32),
        "decoder_action_time_mlp_in_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_fused_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_action_fused_time_biases": torch.zeros((10, 1024), device=device, dtype=torch.float32),
        "decoder_action_mlp_w": torch.zeros((1024, 1024), device=device, dtype=torch.float32),
        "decoder_action_mlp_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_attn_qkv_w": torch.zeros((layer_count, 1024, 2560), device=device, dtype=torch.float32),
        "decoder_attn_o_w": torch.zeros((layer_count, 2048, 1024), device=device, dtype=torch.float32),
        "decoder_ffn_gate_w": torch.zeros((layer_count, 1024, 4096), device=device, dtype=torch.float32),
        "decoder_ffn_up_w": torch.zeros((layer_count, 1024, 4096), device=device, dtype=torch.float32),
        "decoder_ffn_down_w": torch.zeros((layer_count, 4096, 1024), device=device, dtype=torch.float32),
        "decoder_action_fused_out_proj_w": torch.zeros((1024, 32), device=device, dtype=torch.float32),
        "decoder_action_fused_out_proj_b": torch.zeros((32,), device=device, dtype=torch.float32),
    }

    class FullRuntime:
        pass

    full_runtime = FullRuntime()
    full_runtime.weights = weights
    cache = FullCacheSnapshot(
        encoder_seq_len=514,
        encoder_x=None,
        encoder_k=torch.zeros((layer_count, 514, 256), device=device, dtype=torch.float32),
        encoder_v=torch.zeros((layer_count, 514, 256), device=device, dtype=torch.float32),
    )
    kwargs = {
        "cache_snapshot": cache,
        "observation_images_normalized": torch.zeros((2, 224, 224, 3), device=device, dtype=torch.float32),
        "observation_state_normalized": state,
        "noise": torch.zeros((1, 50, 32), device=device, dtype=torch.float32),
        "x0_draft": x0,
        "t_list": (0.10, 0.05),
        "tau_radius": 0.3,
        "dist_dims": 7,
        "max_exec_steps": 12,
        "last_gripper": torch.tensor([-1.0], device=device, dtype=torch.float32),
        "gripper_switch_threshold": 0.0,
        "enable_gripper_verify": True,
        "enable_gripper_post_verify": True,
        "full_runtime": full_runtime,
    }

    fast_result = runtime.run_verify_semantics(**kwargs)
    original = runtime._verify_fast_path_supported
    runtime._verify_fast_path_supported = staticmethod(lambda **_kwargs: False)
    try:
        generic_result = runtime.run_verify_semantics(**kwargs)
    finally:
        runtime._verify_fast_path_supported = original

    assert torch.equal(fast_result.accepted_prefix_len, torch.zeros((1,), device=device, dtype=torch.int64))
    assert torch.equal(fast_result.accepted_prefix_len, generic_result.accepted_prefix_len)
    assert torch.equal(fast_result.metrics, generic_result.metrics)
    assert torch.equal(fast_result.actions, generic_result.actions)
    assert torch.equal(fast_result.x0_hat, generic_result.x0_hat)
    assert float(fast_result.metrics[3].item()) == 1.0
    assert float(fast_result.metrics[4].item()) == 1.0


def test_spec_runtime_verify_fast_x0_hat_matches_fp32_reference(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return

    monkeypatch.setenv("SPEC_TRITON_VERIFY_EXACT_FP32", "1")
    case = _build_verify_exact_fp32_case()
    reference = _compute_verify_reference_outputs(case)

    assert case.runtime._verify_fast_path_supported(
        weights=case.weights,
        cache_snapshot=case.cache,
        x_t_bk=reference.x_t_bk,
    )

    x0_hat = case.runtime.run_verify(
        case.cache,
        torch.zeros((2, 224, 224, 3), device=case.device, dtype=torch.float32),
        case.state,
        case.noise,
        case.x0_draft,
        case.t_list,
        full_runtime=case.full_runtime,
    )

    max_diff = float((x0_hat - reference.x0_hat).abs().max().item())
    assert max_diff < 1.2e-2, f"verify fast x0_hat drift too large: max_diff={max_diff}"

    result = case.runtime.run_verify_semantics(
        cache_snapshot=case.cache,
        observation_images_normalized=torch.zeros((2, 224, 224, 3), device=case.device, dtype=torch.float32),
        observation_state_normalized=case.state,
        noise=case.noise,
        x0_draft=case.x0_draft,
        t_list=case.t_list,
        tau_radius=0.3,
        dist_dims=7,
        max_exec_steps=12,
        last_gripper=None,
        gripper_switch_threshold=0.0,
        enable_gripper_verify=False,
        enable_gripper_post_verify=False,
        full_runtime=case.full_runtime,
    )

    assert torch.equal(result.accepted_prefix_len, reference.accepted_prefix_len)
    assert torch.allclose(result.metrics, reference.metrics, atol=5e-4, rtol=0.0)
    assert torch.allclose(result.actions, reference.actions, atol=1.2e-2, rtol=0.0)


def test_spec_runtime_verify_generic_x0_hat_matches_fp32_reference(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return

    monkeypatch.setenv("SPEC_TRITON_VERIFY_EXACT_FP32", "1")
    case = _build_verify_exact_fp32_case()
    reference = _compute_verify_reference_outputs(case)
    original = case.runtime._verify_fast_path_supported
    case.runtime._verify_fast_path_supported = staticmethod(lambda **_kwargs: False)
    try:
        x0_hat = case.runtime.run_verify(
            case.cache,
            torch.zeros((2, 224, 224, 3), device=case.device, dtype=torch.float32),
            case.state,
            case.noise,
            case.x0_draft,
            case.t_list,
            full_runtime=case.full_runtime,
        )
        result = case.runtime.run_verify_semantics(
            cache_snapshot=case.cache,
            observation_images_normalized=torch.zeros((2, 224, 224, 3), device=case.device, dtype=torch.float32),
            observation_state_normalized=case.state,
            noise=case.noise,
            x0_draft=case.x0_draft,
            t_list=case.t_list,
            tau_radius=0.3,
            dist_dims=7,
            max_exec_steps=12,
            last_gripper=None,
            gripper_switch_threshold=0.0,
            enable_gripper_verify=False,
            enable_gripper_post_verify=False,
            full_runtime=case.full_runtime,
        )
    finally:
        case.runtime._verify_fast_path_supported = original

    max_diff = float((x0_hat - reference.x0_hat).abs().max().item())
    assert max_diff < 5e-3, f"verify generic x0_hat drift too large: max_diff={max_diff}"
    assert torch.equal(result.accepted_prefix_len, reference.accepted_prefix_len)
    assert torch.allclose(result.metrics, reference.metrics, atol=5e-4, rtol=0.0)
    assert torch.allclose(result.actions, reference.actions, atol=5e-3, rtol=0.0)


def test_spec_runtime_verify_graph_matches_plain_fast_path(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return

    monkeypatch.delenv("SPEC_TRITON_VERIFY_EXACT_FP32", raising=False)
    monkeypatch.delenv("SPEC_TRITON_VERIFY_EXACT_INPUT", raising=False)
    case = _build_verify_exact_fp32_case()
    common = {
        "cache_snapshot": case.cache,
        "observation_images_normalized": torch.zeros((2, 224, 224, 3), device=case.device, dtype=torch.float32),
        "observation_state_normalized": case.state,
        "noise": case.noise,
        "x0_draft": case.x0_draft,
        "t_list": case.t_list,
        "tau_radius": 0.3,
        "dist_dims": 7,
        "max_exec_steps": 12,
        "last_gripper": None,
        "gripper_switch_threshold": 0.0,
        "enable_gripper_verify": False,
        "enable_gripper_post_verify": False,
        "full_runtime": case.full_runtime,
    }

    monkeypatch.setenv("SPEC_TRITON_VERIFY_GRAPH", "0")
    plain = case.runtime.run_verify_semantics(**common)
    monkeypatch.setenv("SPEC_TRITON_VERIFY_GRAPH", "1")
    graph = case.runtime.run_verify_semantics(**common)

    assert case.runtime._verify_fast_graph is not None
    assert not case.runtime._verify_fast_graph_failed
    assert torch.equal(graph.x0_hat, plain.x0_hat)
    assert torch.equal(graph.actions, plain.actions)
    assert torch.equal(graph.metrics, plain.metrics)
    assert torch.equal(graph.accepted_prefix_len, plain.accepted_prefix_len)


def test_spec_runtime_verify_semantics_graph_matches_plain_and_recaptures_max_exec(monkeypatch) -> None:
    if not torch.cuda.is_available():
        return

    monkeypatch.delenv("SPEC_TRITON_VERIFY_EXACT_FP32", raising=False)
    monkeypatch.delenv("SPEC_TRITON_VERIFY_EXACT_INPUT", raising=False)
    monkeypatch.setenv("SPEC_TRITON_POSTPROCESS_FUSED", "1")
    case = _build_verify_exact_fp32_case()
    for key, value in list(case.weights.items()):
        case.weights[key] = value.to(device=case.device, dtype=torch.bfloat16).contiguous()
    case.cache = dataclasses.replace(
        case.cache,
        encoder_k=case.cache.encoder_k.to(dtype=torch.bfloat16).contiguous(),
        encoder_v=case.cache.encoder_v.to(dtype=torch.bfloat16).contiguous(),
    )
    common = {
        "cache_snapshot": case.cache,
        "observation_images_normalized": torch.zeros((2, 224, 224, 3), device=case.device, dtype=torch.float32),
        "observation_state_normalized": case.state,
        "noise": case.noise,
        "x0_draft": case.x0_draft,
        "t_list": case.t_list,
        "tau_radius": 0.3,
        "dist_dims": 7,
        "max_exec_steps": 12,
        "last_gripper": None,
        "gripper_switch_threshold": 0.0,
        "enable_gripper_verify": False,
        "enable_gripper_post_verify": False,
        "full_runtime": case.full_runtime,
    }

    monkeypatch.setenv("SPEC_TRITON_VERIFY_GRAPH", "0")
    plain, _ = case.runtime.run_verify_semantics_with_timing(**common)
    monkeypatch.setenv("SPEC_TRITON_VERIFY_GRAPH", "1")
    graphed, _ = case.runtime.run_verify_semantics_with_timing(**common)

    assert case.runtime._verify_semantics_graph is not None
    assert not case.runtime._verify_semantics_graph_failed
    first_key = case.runtime._verify_semantics_graph.key
    assert torch.equal(graphed.x0_hat, plain.x0_hat)
    assert torch.equal(graphed.actions, plain.actions)
    assert torch.equal(graphed.metrics, plain.metrics)
    assert torch.equal(graphed.accepted_prefix_len, plain.accepted_prefix_len)

    common = {**common, "max_exec_steps": 5}
    monkeypatch.setenv("SPEC_TRITON_VERIFY_GRAPH", "0")
    plain_small, _ = case.runtime.run_verify_semantics_with_timing(**common)
    monkeypatch.setenv("SPEC_TRITON_VERIFY_GRAPH", "1")
    graphed_small, _ = case.runtime.run_verify_semantics_with_timing(**common)

    assert case.runtime._verify_semantics_graph is not None
    assert case.runtime._verify_semantics_graph.key != first_key
    assert torch.equal(graphed_small.x0_hat, plain_small.x0_hat)
    assert torch.equal(graphed_small.actions, plain_small.actions)
    assert torch.equal(graphed_small.metrics, plain_small.metrics)
    assert torch.equal(graphed_small.accepted_prefix_len, plain_small.accepted_prefix_len)


def test_spec_verify_query0_mask_keeps_state_token_visible() -> None:
    if not torch.cuda.is_available():
        return

    module_path = Path(__file__).resolve().parents[1] / "triton" / "pi0_spec_infer.py"
    spec = importlib.util.spec_from_file_location("pi0_spec_infer_mask_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    device = torch.device("cuda:0")
    logits = torch.zeros((2, 5), device=device, dtype=torch.float32)
    probs = torch.empty((2, 5), device=device, dtype=torch.bfloat16)
    key_valid = torch.ones((1, 5), device=device, dtype=torch.int8)

    module._attention_softmax_mask_kernel[(1,)](
        logits,
        key_valid,
        probs,
        total_rows=2,
        total_keys=5,
        num_heads=1,
        query_len=2,
        prefix_len=3,
        block_suffix_for_query0=1,
        BLOCK_ROWS=4,
        BLOCK_SIZE=8,
    )
    probs_f = probs.to(dtype=torch.float32)
    assert probs_f[0, 3] > 0.0
    assert probs_f[0, 4] == 0.0
    assert probs_f[1, 4] > 0.0

    grouped_probs = torch.empty((2, 5), device=device, dtype=torch.bfloat16)
    module._grouped_softmax_mask0_kernel[(1,)](
        logits,
        grouped_probs,
        total_queries=2,
        keys_per_group=5,
        num_groups=1,
        num_heads=1,
        query_len=2,
        prefix_len=3,
        block_suffix_for_query0=1,
        BLOCK_ROWS=4,
        BLOCK_SIZE=8,
    )
    grouped_probs_f = grouped_probs.to(dtype=torch.float32)
    assert grouped_probs_f[0, 3] > 0.0
    assert grouped_probs_f[0, 4] == 0.0
    assert grouped_probs_f[1, 4] > 0.0


def test_spec_runtime_builds_bk_verify_batch(tmp_path: Path) -> None:
    session = _build_spec_session(tmp_path)
    runtime = session.draft_runtime

    x_t_bk, timestep_bk = runtime._build_verify_batch(
        noise=torch.zeros((1, 50, 32), dtype=torch.float32),
        x0_draft=torch.ones((1, 50, 32), dtype=torch.float32),
        t_list=(0.10, 0.05),
    )

    assert x_t_bk.shape == (2, 50, 32)
    assert timestep_bk.shape == (2,)
    assert torch.allclose(timestep_bk, torch.tensor([0.10, 0.05], dtype=torch.float32))


def test_spec_runtime_verify_expands_time_bias_per_action_row(monkeypatch) -> None:
    module_path = Path(__file__).resolve().parents[1] / "triton" / "pi0_spec_infer.py"
    spec = importlib.util.spec_from_file_location("pi0_spec_infer_time_bias_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    Pi0SpecInference = module.Pi0SpecInference
    FullCacheSnapshot = module.FullCacheSnapshot
    PrefixContext = module.PrefixContext

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = {"language_embeds": torch.zeros((2, 2048), dtype=torch.bfloat16)}
    draft = {
        "meta": {"img_dim": 2048, "chunk_m": 50, "out_dim": 7, "draft_num_heads": 8, "draft_num_kv_heads": 1, "draft_head_dim": 256},
        "draft_state_in_proj_w": torch.zeros((2048, 32), dtype=torch.float32),
        "draft_state_in_proj_b": torch.zeros((2048,), dtype=torch.float32),
        "draft_action_queries": torch.zeros((50, 2048), dtype=torch.float32),
        "draft_qkv_w": torch.zeros((2560, 2048), dtype=torch.float32),
        "draft_attn_o_w": torch.zeros((2048, 2048), dtype=torch.float32),
        "draft_ffn_gate_w": torch.zeros((16384, 2048), dtype=torch.float32),
        "draft_ffn_up_w": torch.zeros((16384, 2048), dtype=torch.float32),
        "draft_ffn_down_w": torch.zeros((2048, 16384), dtype=torch.float32),
        "draft_input_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
        "draft_post_attention_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
        "draft_action_head_w": torch.zeros((7, 2048), dtype=torch.float32),
        "draft_action_head_b": torch.zeros((7,), dtype=torch.float32),
    }
    runtime = Pi0SpecInference(checkpoint=checkpoint, draft_checkpoint=draft, num_views=2, chunk_size=50)

    original = Pi0SpecInference._add_time_bias_silu
    observed: dict[str, tuple[int, int]] = {}

    def checked_add_time_bias_silu(self, *, x, bias):
        observed["x_shape"] = tuple(x.shape)
        observed["bias_shape"] = tuple(bias.shape)
        assert int(bias.shape[0]) == int(x.shape[0]), f"bias rows {tuple(bias.shape)} must match x rows {tuple(x.shape)}"
        return original(self, x=x, bias=bias)

    monkeypatch.setattr(Pi0SpecInference, "_add_time_bias_silu", checked_add_time_bias_silu, raising=True)

    prefix = PrefixContext(
        prefix_embs=torch.zeros((1, 514, 2048), device=device, dtype=torch.float32),
        prefix_pad_masks=torch.ones((1, 514), device=device, dtype=torch.bool),
        prefix_att_masks=torch.zeros((1, 514), device=device, dtype=torch.bool),
    )
    state = torch.zeros((32,), device=device, dtype=torch.float32)
    x0 = runtime._run_draft_block(prefix=prefix, observation_state_normalized=state)
    layer_count = 1
    weights = {
        "decoder_state_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_state_in_proj_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_action_in_proj_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_time_mlp_in_w": torch.zeros((2048, 1024), device=device, dtype=torch.float32),
        "decoder_action_time_mlp_in_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_fused_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_action_fused_time_biases": torch.zeros((10, 1024), device=device, dtype=torch.float32),
        "decoder_action_mlp_w": torch.zeros((1024, 1024), device=device, dtype=torch.float32),
        "decoder_action_mlp_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_attn_qkv_w": torch.zeros((layer_count, 1024, 2560), device=device, dtype=torch.float32),
        "decoder_attn_o_w": torch.zeros((layer_count, 2048, 1024), device=device, dtype=torch.float32),
        "decoder_ffn_gate_w": torch.zeros((layer_count, 1024, 4096), device=device, dtype=torch.float32),
        "decoder_ffn_up_w": torch.zeros((layer_count, 1024, 4096), device=device, dtype=torch.float32),
        "decoder_ffn_down_w": torch.zeros((layer_count, 4096, 1024), device=device, dtype=torch.float32),
        "decoder_action_fused_out_proj_w": torch.zeros((1024, 32), device=device, dtype=torch.float32),
        "decoder_action_fused_out_proj_b": torch.zeros((32,), device=device, dtype=torch.float32),
    }

    class FullRuntime:
        pass

    full_runtime = FullRuntime()
    full_runtime.weights = weights
    cache = FullCacheSnapshot(
        encoder_seq_len=514,
        encoder_x=None,
        encoder_k=torch.zeros((layer_count, 514, 256), device=device, dtype=torch.float32),
        encoder_v=torch.zeros((layer_count, 514, 256), device=device, dtype=torch.float32),
    )

    result = runtime.run_verify_semantics(
        cache_snapshot=cache,
        observation_images_normalized=torch.zeros((2, 224, 224, 3), device=device, dtype=torch.float32),
        observation_state_normalized=state,
        noise=torch.zeros((1, 50, 32), device=device, dtype=torch.float32),
        x0_draft=x0,
        t_list=(0.10, 0.05),
        tau_radius=0.3,
        dist_dims=7,
        max_exec_steps=12,
        last_gripper=None,
        gripper_switch_threshold=0.0,
        enable_gripper_verify=True,
        enable_gripper_post_verify=True,
        full_runtime=full_runtime,
    )

    assert result.actions.shape == (1, 50, 32)
    assert observed["x_shape"] == (100, 1024)
    assert observed["bias_shape"] == (100, 1024)


def test_spec_runtime_verify_uses_exact_time_mlp_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SPEC_TRITON_VERIFY_EXACT_INPUT", "1")
    module_path = Path(__file__).resolve().parents[1] / "triton" / "pi0_spec_infer.py"
    spec = importlib.util.spec_from_file_location("pi0_spec_infer_exact_verify_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    Pi0SpecInference = module.Pi0SpecInference
    FullCacheSnapshot = module.FullCacheSnapshot
    PrefixContext = module.PrefixContext

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = {"language_embeds": torch.zeros((2, 2048), dtype=torch.bfloat16)}
    draft = {
        "meta": {"img_dim": 2048, "chunk_m": 50, "out_dim": 7, "draft_num_heads": 8, "draft_num_kv_heads": 1, "draft_head_dim": 256},
        "draft_state_in_proj_w": torch.zeros((2048, 32), dtype=torch.float32),
        "draft_state_in_proj_b": torch.zeros((2048,), dtype=torch.float32),
        "draft_action_queries": torch.zeros((50, 2048), dtype=torch.float32),
        "draft_qkv_w": torch.zeros((2560, 2048), dtype=torch.float32),
        "draft_attn_o_w": torch.zeros((2048, 2048), dtype=torch.float32),
        "draft_ffn_gate_w": torch.zeros((16384, 2048), dtype=torch.float32),
        "draft_ffn_up_w": torch.zeros((16384, 2048), dtype=torch.float32),
        "draft_ffn_down_w": torch.zeros((2048, 16384), dtype=torch.float32),
        "draft_input_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
        "draft_post_attention_layernorm_w": torch.ones((2048,), dtype=torch.bfloat16),
        "draft_action_head_w": torch.zeros((7, 2048), dtype=torch.float32),
        "draft_action_head_b": torch.zeros((7,), dtype=torch.float32),
    }
    runtime = Pi0SpecInference(checkpoint=checkpoint, draft_checkpoint=draft, num_views=2, chunk_size=50)

    def fail_time_bias(*args, **kwargs):
        raise AssertionError("exact verify path should not use fused time bias interpolation")

    monkeypatch.setattr(runtime, "_time_bias", fail_time_bias, raising=True)

    prefix = PrefixContext(
        prefix_embs=torch.zeros((1, 514, 2048), device=device, dtype=torch.float32),
        prefix_pad_masks=torch.ones((1, 514), device=device, dtype=torch.bool),
        prefix_att_masks=torch.zeros((1, 514), device=device, dtype=torch.bool),
    )
    state = torch.zeros((32,), device=device, dtype=torch.float32)
    x0 = runtime._run_draft_block(prefix=prefix, observation_state_normalized=state)
    layer_count = 18
    weights = {
        "decoder_state_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_state_in_proj_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_action_in_proj_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_time_mlp_in_w": torch.zeros((2048, 1024), device=device, dtype=torch.float32),
        "decoder_action_time_mlp_in_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_action_fused_in_proj_w": torch.zeros((32, 1024), device=device, dtype=torch.float32),
        "decoder_action_fused_time_biases": torch.zeros((10, 1024), device=device, dtype=torch.float32),
        "decoder_action_mlp_w": torch.zeros((1024, 1024), device=device, dtype=torch.float32),
        "decoder_action_mlp_b": torch.zeros((1024,), device=device, dtype=torch.float32),
        "decoder_attn_qkv_w": torch.zeros((layer_count, 1024, 2560), device=device, dtype=torch.float32),
        "decoder_attn_o_w": torch.zeros((layer_count, 2048, 1024), device=device, dtype=torch.float32),
        "decoder_ffn_gate_w": torch.zeros((layer_count, 1024, 4096), device=device, dtype=torch.float32),
        "decoder_ffn_up_w": torch.zeros((layer_count, 1024, 4096), device=device, dtype=torch.float32),
        "decoder_ffn_down_w": torch.zeros((layer_count, 4096, 1024), device=device, dtype=torch.float32),
        "decoder_action_fused_out_proj_w": torch.zeros((1024, 32), device=device, dtype=torch.float32),
        "decoder_action_fused_out_proj_b": torch.zeros((32,), device=device, dtype=torch.float32),
    }

    class FullRuntime:
        pass

    full_runtime = FullRuntime()
    full_runtime.weights = weights
    cache = FullCacheSnapshot(
        encoder_seq_len=514,
        encoder_x=None,
        encoder_k=torch.zeros((layer_count, 514, 256), device=device, dtype=torch.float32),
        encoder_v=torch.zeros((layer_count, 514, 256), device=device, dtype=torch.float32),
    )

    result = runtime.run_verify_semantics(
        cache_snapshot=cache,
        observation_images_normalized=torch.zeros((2, 224, 224, 3), device=device, dtype=torch.float32),
        observation_state_normalized=state,
        noise=torch.zeros((1, 50, 32), device=device, dtype=torch.float32),
        x0_draft=x0,
        t_list=(0.10, 0.05),
        tau_radius=0.3,
        dist_dims=7,
        max_exec_steps=12,
        last_gripper=None,
        gripper_switch_threshold=0.0,
        enable_gripper_verify=True,
        enable_gripper_post_verify=True,
        full_runtime=full_runtime,
    )

    assert result.actions.shape == (1, 50, 32)


def test_spec_runtime_prepare_prefix_reuses_encoder_without_running_prefill(tmp_path: Path) -> None:
    base_weights_path, manifest_path, draft_triton_path = _write_small_spec_session_artifacts(tmp_path)
    observed = {"encoder": 0, "prefill": 0}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class FakeFullRuntime:
        def __init__(self, prompt_len: int):
            self.weights = {
                "language_embeds": torch.ones((prompt_len, 8), dtype=torch.float32, device=device),
                "vision_final_norm_w": torch.ones((8,), dtype=torch.float32, device=device),
                "vision_final_norm_b": torch.zeros((8,), dtype=torch.float32, device=device),
                "encoder_multi_modal_projector_w": torch.eye(8, dtype=torch.float32, device=device),
                "encoder_multi_modal_projector_b": torch.zeros((8,), dtype=torch.float32, device=device),
            }
            self.buffers = {
                "observation_images_normalized": torch.zeros((2, 4, 4, 3), dtype=torch.float32, device=device),
                "observation_state_normalized": torch.zeros((32,), dtype=torch.float32, device=device),
                "vision_x": torch.zeros((2, 1, 8), dtype=torch.float32, device=device),
                "vision_x_norm": torch.zeros((2, 1, 8), dtype=torch.float32, device=device),
            }
            self._encoder_graph = object()
            self._prefill_graph = object()
            self._encoder_seq_len = 4

        def _record_encoder_stage(self):
            observed["encoder"] += 1
            self.buffers["vision_x"].fill_(2.0)

        def _record_prefill_stage(self):
            observed["prefill"] += 1
            raise AssertionError("draft prefix preparation should not run full prefill")

        def _replay_or_run(self, graph, record_fn):
            del graph
            record_fn()
            return 0.0

    pool = triton_runtime.SpecTritonRuntimePool(
        base_weights_path=base_weights_path,
        manifest_path=manifest_path,
        draft_checkpoint_path=draft_triton_path,
        num_views=2,
        chunk_size=50,
        runtime_factory=lambda *, checkpoint, **_kwargs: FakeFullRuntime(int(checkpoint["language_embeds"].shape[0])),
    )

    session = pool.start_session("task a")
    prepared = session.prepare_observation(
        images=torch.zeros((2, 4, 4, 3), dtype=torch.float32),
        state=torch.zeros((32,), dtype=torch.float32),
    )

    x0_draft, timing = session.run_draft_with_timing(
        prepared=prepared,
    )

    assert x0_draft.shape == (1, 50, 32)
    assert timing["encoder_ms"] >= 0.0
    assert observed == {"encoder": 1, "prefill": 0}


def test_spec_policy_runtime_tracks_last_gripper_without_last_actions_history() -> None:
    action_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    observed: dict[str, object] = {"run_full": 0, "run_draft": 0}

    class FakeSession:
        def prepare_observation(self, *, images, state):
            return SimpleNamespace(images=images, state=state)

        def run_full_with_timing(self, *, prepared, noise):
            del prepared, noise
            observed["run_full"] = int(observed["run_full"]) + 1
            return torch.ones((50, 32), dtype=torch.float32, device=action_device), {
                "encoder_ms": 1.0,
                "vlm_prefill_ms": 2.0,
                "decoder_ms": 3.0,
                "total_ms": 6.0,
            }

        def capture_full_cache_snapshot(self):
            return "snapshot-1"

        def run_draft_with_timing(self, *, prepared):
            del prepared
            observed["run_draft"] = int(observed["run_draft"]) + 1
            return torch.zeros((1, 50, 32), dtype=torch.float32, device=action_device), {
                "encoder_ms": 4.0,
                "draft_ms": 5.0,
            }

        def run_verify_semantics_with_timing(self, *, cache_snapshot, prepared, noise, x0_draft, t_list, **kwargs):
            del prepared, noise, x0_draft, t_list
            observed["verify_snapshot"] = cache_snapshot
            observed["last_gripper"] = kwargs["last_gripper"].detach().cpu().tolist()
            observed["last_gripper_device"] = str(kwargs["last_gripper"].device)
            return (
                SimpleNamespace(
                    actions=torch.zeros((1, 50, 32), dtype=torch.float32, device=action_device),
                    metrics=torch.tensor([0.1, 5.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=action_device),
                    accepted_prefix_len=torch.tensor([5], dtype=torch.int64, device=action_device),
                ),
                {"action_verify_ms": 6.0},
            )

    class FakeRuntimePool:
        def start_session(self, prompt: str):
            observed["prompt"] = prompt
            return FakeSession()

    runtime = triton_runtime.SpecTritonPolicyRuntime(
        runtime_pool=FakeRuntimePool(),
        action_horizon=50,
        action_dim=32,
        max_exec_steps=12,
        device="cpu",
    )
    images = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
    state = torch.zeros((8,), dtype=torch.float32)
    noise = torch.zeros((50, 32), dtype=torch.float32)

    first_actions, first_timing = runtime.sample_actions_with_timing(
        prompt="task a",
        images=images,
        state=state,
        noise=noise,
    )
    second_actions, second_timing = runtime.sample_actions_with_timing(
        prompt="task a",
        images=images,
        state=state,
        noise=noise,
        executed_steps=1,
    )

    assert first_actions.shape == (50, 32)
    assert first_timing["is_full_pipeline_round"] == 1.0
    assert second_actions.shape == (50, 32)
    assert second_timing["is_full_pipeline_round"] == 0.0
    assert observed["run_full"] == 1
    assert observed["run_draft"] == 1
    assert observed["verify_snapshot"] == "snapshot-1"
    assert observed["last_gripper"] == [1.0]
    assert observed["last_gripper_device"] == str(action_device)


def test_compiled_spec_verify_runtime_matches_fp32_reference() -> None:
    class FakeSpecModel:
        @staticmethod
        def _signal_from_cache(key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return (
                key[0, 0, 0, 0].to(dtype=torch.float32) * 7.0
                - key[0, 0, 1, 1].to(dtype=torch.float32) * 3.0
                + value[0, 0, 0, 1].to(dtype=torch.float32) * 11.0
                + value[0, 0, 1, 0].to(dtype=torch.float32) * 5.0
            )

        def _action_stage(self, state, prefix_pad_masks, past_key_values, noise, x0_draft, last_gripper):
            key, value = past_key_values[0]
            signal = self._signal_from_cache(key, value)
            signal = signal + state.to(dtype=torch.float32).mean() * 0.1
            signal = signal + noise.to(dtype=torch.float32).mean() * 0.01
            if last_gripper is not None:
                signal = signal + last_gripper.to(dtype=torch.float32).mean() * 0.25
            actions = x0_draft.to(dtype=torch.float32) + signal.view(1, 1, 1)
            metrics = torch.stack(
                (
                    signal,
                    actions.mean(),
                    actions.abs().max(),
                    prefix_pad_masks.to(dtype=torch.float32).mean(),
                    x0_draft.to(dtype=torch.float32).mean(),
                )
            )
            accepted_prefix_len = torch.full(
                (int(state.shape[0]),),
                min(5, int(x0_draft.shape[1])),
                dtype=torch.int64,
                device=x0_draft.device,
            )
            return actions, metrics, accepted_prefix_len

    def _reference_outputs(*, cache_snapshot, state, prefix_pad_masks, noise, x0_draft, last_gripper):
        signal = FakeSpecModel._signal_from_cache(
            cache_snapshot.encoder_k[:, : cache_snapshot.encoder_seq_len].unsqueeze(1),
            cache_snapshot.encoder_v[:, : cache_snapshot.encoder_seq_len].unsqueeze(1),
        )
        signal = signal + state.mean() * 0.1
        signal = signal + noise.mean() * 0.01
        if last_gripper is not None:
            signal = signal + last_gripper.mean() * 0.25
        actions = x0_draft + signal.view(1, 1, 1)
        metrics = torch.stack(
            (
                signal,
                actions.mean(),
                actions.abs().max(),
                prefix_pad_masks.to(dtype=torch.float32).mean(),
                x0_draft.mean(),
            )
        )
        accepted_prefix_len = torch.full(
            (int(state.shape[0]),),
            min(5, int(x0_draft.shape[1])),
            dtype=torch.int64,
            device=x0_draft.device,
        )
        return actions, metrics, accepted_prefix_len

    runtime = triton_runtime.CompiledSpecVerifyRuntime(spec_model=FakeSpecModel(), device='cpu')
    cache_snapshot = SimpleNamespace(
        encoder_seq_len=3,
        encoder_k=torch.tensor(
            [[[1.0010, 1.0020], [1.0030, 1.0040], [1.0050, 1.0060]]],
            dtype=torch.float32,
        ),
        encoder_v=torch.tensor(
            [[[0.2510, 0.2520], [0.2530, 0.2540], [0.2550, 0.2560]]],
            dtype=torch.float32,
        ),
    )
    prepared = SimpleNamespace(state=torch.tensor([0.1250, -0.2500, 0.5000], dtype=torch.float32))
    noise = torch.tensor([[0.10, -0.20], [0.30, -0.40]], dtype=torch.float32)
    x0_draft = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    last_gripper = torch.tensor([1.0], dtype=torch.float32)

    result, timing = runtime.run_verify_semantics_with_timing(
        cache_snapshot=cache_snapshot,
        prepared=prepared,
        noise=noise,
        x0_draft=x0_draft,
        t_list=(0.10, 0.05),
        tau_radius=0.3,
        dist_dims=7,
        max_exec_steps=12,
        last_gripper=last_gripper,
        gripper_switch_threshold=0.0,
        enable_gripper_verify=True,
        enable_gripper_post_verify=True,
    )

    state_ref = runtime._as_state_batch(prepared.state).to(dtype=torch.float32)
    noise_ref = runtime._as_action_batch(noise).to(dtype=torch.float32)
    x0_draft_ref = runtime._as_action_batch(x0_draft).to(dtype=torch.float32)
    prefix_pad_masks = torch.ones((1, cache_snapshot.encoder_seq_len), dtype=torch.bool)
    expected_actions, expected_metrics, expected_prefix = _reference_outputs(
        cache_snapshot=cache_snapshot,
        state=state_ref,
        prefix_pad_masks=prefix_pad_masks,
        noise=noise_ref,
        x0_draft=x0_draft_ref,
        last_gripper=last_gripper.to(dtype=torch.float32),
    )

    assert torch.equal(result.accepted_prefix_len, expected_prefix)
    assert torch.allclose(result.metrics, expected_metrics, atol=1e-6, rtol=0.0)
    assert torch.allclose(result.actions, expected_actions, atol=1e-6, rtol=0.0)
    assert timing["action_verify_ms"] >= 0.0


def test_spec_policy_runtime_prefers_compiled_verify_backend() -> None:
    action_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    observed: dict[str, object] = {"compiled_verify_calls": 0}

    class FakeSession:
        def prepare_observation(self, *, images, state):
            return SimpleNamespace(images=images, state=state)

        def run_full_with_timing(self, *, prepared, noise):
            del prepared, noise
            return torch.ones((50, 32), dtype=torch.float32, device=action_device), {
                "encoder_ms": 1.0,
                "vlm_prefill_ms": 2.0,
                "decoder_ms": 3.0,
                "total_ms": 6.0,
            }

        def capture_full_cache_snapshot(self):
            return "snapshot-1"

        def run_draft_with_timing(self, *, prepared):
            del prepared
            return torch.zeros((1, 50, 32), dtype=torch.float32, device=action_device), {
                "encoder_ms": 4.0,
                "draft_ms": 5.0,
            }

        def run_verify_semantics_with_timing(self, **_kwargs):
            raise AssertionError("compiled verify backend should bypass Triton verify")

    class FakeRuntimePool:
        def start_session(self, prompt: str):
            observed["prompt"] = prompt
            return FakeSession()

    class FakeCompiledVerify:
        def run_verify_semantics_with_timing(
            self,
            *,
            cache_snapshot,
            prepared,
            noise,
            x0_draft,
            t_list,
            tau_radius,
            dist_dims,
            max_exec_steps,
            last_gripper,
            gripper_switch_threshold,
            enable_gripper_verify,
            enable_gripper_post_verify,
        ):
            observed["compiled_verify_calls"] = int(observed["compiled_verify_calls"]) + 1
            observed["cache_snapshot"] = cache_snapshot
            observed["prepared_state_shape"] = tuple(prepared.state.shape)
            observed["noise_shape"] = tuple(noise.shape)
            observed["x0_draft_shape"] = tuple(x0_draft.shape)
            observed["t_list"] = tuple(float(x) for x in t_list)
            observed["last_gripper"] = None if last_gripper is None else last_gripper.detach().cpu().tolist()
            observed["flags"] = (
                float(tau_radius),
                int(dist_dims),
                int(max_exec_steps),
                float(gripper_switch_threshold),
                bool(enable_gripper_verify),
                bool(enable_gripper_post_verify),
            )
            return (
                SimpleNamespace(
                    actions=torch.full((1, 50, 32), 7.0, dtype=torch.float32, device=action_device),
                    metrics=torch.tensor([0.1, 5.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=action_device),
                    accepted_prefix_len=torch.tensor([5], dtype=torch.int64, device=action_device),
                ),
                {"action_verify_ms": 6.0},
            )

    runtime = triton_runtime.SpecTritonPolicyRuntime(
        runtime_pool=FakeRuntimePool(),
        action_horizon=50,
        action_dim=32,
        max_exec_steps=12,
        device="cpu",
        compiled_verify_runtime=FakeCompiledVerify(),
    )
    images = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
    state = torch.zeros((8,), dtype=torch.float32)
    noise = torch.zeros((50, 32), dtype=torch.float32)

    runtime.sample_actions_with_timing(
        prompt="task a",
        images=images,
        state=state,
        noise=noise,
    )
    second_actions, second_timing = runtime.sample_actions_with_timing(
        prompt="task a",
        images=images,
        state=state,
        noise=noise,
        executed_steps=1,
    )

    assert second_actions.shape == (50, 32)
    assert torch.all(second_actions == 7.0)
    assert second_timing["accepted_prefix_len"] == 5.0
    assert observed["compiled_verify_calls"] == 1
    assert observed["cache_snapshot"] == "snapshot-1"
    assert observed["prepared_state_shape"] == (8,)
    assert observed["noise_shape"] == (1, 50, 32)
    assert observed["x0_draft_shape"] == (1, 50, 32)
    assert observed["t_list"] == (0.10, 0.05)
    assert observed["last_gripper"] == [1.0]
    assert observed["flags"] == (0.3, 7, 12, 0.0, True, True)


def test_spec_policy_runtime_prefers_compiled_draft_backend_and_tracks_last_actions() -> None:
    action_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    observed: dict[str, object] = {"compiled_draft_calls": 0, "compiled_verify_calls": 0}

    class FakeSession:
        def __init__(self) -> None:
            self._runtime = object()
            self.draft_runtime = object()

        def prepare_observation(self, *, images, state):
            return SimpleNamespace(images=images, state=state)

        def run_full_with_timing(self, *, prepared, noise):
            del prepared, noise
            return torch.ones((50, 32), dtype=torch.float32, device=action_device), {
                "encoder_ms": 1.0,
                "vlm_prefill_ms": 2.0,
                "decoder_ms": 3.0,
                "total_ms": 6.0,
            }

        def capture_full_cache_snapshot(self):
            return "snapshot-1"

        def run_draft_with_timing(self, **_kwargs):
            raise AssertionError("compiled draft backend should bypass Triton draft")

        def run_verify_semantics_with_timing(self, **_kwargs):
            raise AssertionError("compiled verify backend should bypass Triton verify")

    class FakeRuntimePool:
        def start_session(self, prompt: str):
            observed["prompt"] = prompt
            return FakeSession()

    class FakeCompiledDraft:
        def run_draft_with_timing(self, *, prepared, noise, last_actions, prefix_runtime, full_runtime):
            del prefix_runtime, full_runtime
            observed["compiled_draft_calls"] = int(observed["compiled_draft_calls"]) + 1
            observed["prepared_state_shape"] = tuple(prepared.state.shape)
            observed["noise_shape"] = tuple(noise.shape)
            observed["last_actions_shape"] = tuple(last_actions.shape)
            observed["last_actions_tail"] = last_actions[:, -1, :].detach().cpu()
            return (
                torch.full((1, 50, 32), 4.0, dtype=torch.float32, device=action_device),
                {"encoder_ms": 4.0, "draft_ms": 5.0},
            )

    class FakeCompiledVerify:
        def run_verify_semantics_with_timing(self, *, x0_draft, **_kwargs):
            observed["compiled_verify_calls"] = int(observed["compiled_verify_calls"]) + 1
            observed["x0_draft_mean"] = float(x0_draft.mean().item())
            return (
                SimpleNamespace(
                    actions=torch.full((1, 50, 32), 7.0, dtype=torch.float32, device=action_device),
                    metrics=torch.tensor([0.1, 5.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=action_device),
                    accepted_prefix_len=torch.tensor([5], dtype=torch.int64, device=action_device),
                ),
                {"action_verify_ms": 6.0},
            )

    runtime = triton_runtime.SpecTritonPolicyRuntime(
        runtime_pool=FakeRuntimePool(),
        action_horizon=50,
        action_dim=32,
        max_exec_steps=12,
        device="cpu",
        compiled_draft_runtime=FakeCompiledDraft(),
        compiled_verify_runtime=FakeCompiledVerify(),
        draft_history_len=3,
    )
    images = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
    state = torch.zeros((8,), dtype=torch.float32)
    noise = torch.zeros((50, 32), dtype=torch.float32)

    runtime.sample_actions_with_timing(
        prompt="task a",
        images=images,
        state=state,
        noise=noise,
    )
    second_actions, second_timing = runtime.sample_actions_with_timing(
        prompt="task a",
        images=images,
        state=state,
        noise=noise,
        executed_steps=1,
    )

    assert second_actions.shape == (50, 32)
    assert torch.all(second_actions == 7.0)
    assert second_timing["accepted_prefix_len"] == 5.0
    assert observed["compiled_draft_calls"] == 1
    assert observed["compiled_verify_calls"] == 1
    assert observed["prepared_state_shape"] == (8,)
    assert observed["noise_shape"] == (1, 50, 32)
    assert observed["last_actions_shape"] == (1, 3, 32)
    assert torch.all(observed["last_actions_tail"] == 1.0)
    assert observed["x0_draft_mean"] == 4.0


def test_spec_policy_runtime_passes_tokenized_prompt_to_compiled_backend() -> None:
    action_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    observed: dict[str, object] = {"compiled_full_calls": 0, "compiled_draft_calls": 0}

    class FakeRuntimePool:
        def start_session(self, prompt: str):
            observed["prompt"] = prompt
            return triton_runtime.TritonRuntimeSession(prompt=prompt, runtime=object())

    class FakeCompiledBackend:
        def __init__(self) -> None:
            self._snapshot = None

        def run_full_with_timing(self, *, prepared, noise, full_runtime):
            del noise, full_runtime
            observed["compiled_full_calls"] = int(observed["compiled_full_calls"]) + 1
            observed["full_tokens"] = prepared.tokenized_prompt.detach().cpu().tolist()
            observed["full_token_mask"] = prepared.tokenized_prompt_mask.detach().cpu().tolist()
            self._snapshot = SimpleNamespace(
                encoder_seq_len=3,
                encoder_k=torch.zeros((1, 3, 2), dtype=torch.float32),
                encoder_v=torch.zeros((1, 3, 2), dtype=torch.float32),
                prefix_pad_masks=torch.tensor([[True, True, False]], dtype=torch.bool),
            )
            return torch.ones((1, 50, 32), dtype=torch.float32, device=action_device), {
                "encoder_ms": 1.0,
                "vlm_prefill_ms": 2.0,
                "decoder_ms": 3.0,
                "total_ms": 6.0,
            }

        def capture_full_cache_snapshot(self):
            return self._snapshot

        def run_draft_with_timing(
            self,
            *,
            prepared,
            noise,
            last_actions,
            prefix_runtime,
            full_runtime,
            encoder_runtime,
        ):
            del noise, last_actions, prefix_runtime, full_runtime
            observed["compiled_draft_calls"] = int(observed["compiled_draft_calls"]) + 1
            observed["draft_tokens"] = prepared.tokenized_prompt.detach().cpu().tolist()
            observed["draft_token_mask"] = prepared.tokenized_prompt_mask.detach().cpu().tolist()
            observed["encoder_runtime_is_self"] = encoder_runtime is self
            return (
                torch.full((1, 50, 32), 4.0, dtype=torch.float32, device=action_device),
                {"encoder_ms": 4.0, "draft_ms": 5.0},
            )

        def run_verify_semantics_with_timing(self, *, x0_draft, **_kwargs):
            observed["x0_draft_mean"] = float(x0_draft.mean().item())
            return (
                SimpleNamespace(
                    actions=torch.full((1, 50, 32), 7.0, dtype=torch.float32, device=action_device),
                    metrics=torch.tensor([0.1, 5.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=action_device),
                    accepted_prefix_len=torch.tensor([5], dtype=torch.int64, device=action_device),
                ),
                {"action_verify_ms": 6.0},
            )

    compiled_backend = FakeCompiledBackend()
    runtime = triton_runtime.SpecTritonPolicyRuntime(
        runtime_pool=FakeRuntimePool(),
        action_horizon=50,
        action_dim=32,
        max_exec_steps=12,
        device="cpu",
        compiled_encoder_runtime=compiled_backend,
        compiled_draft_runtime=compiled_backend,
        compiled_verify_runtime=compiled_backend,
        draft_history_len=3,
    )
    images = torch.zeros((2, 4, 4, 3), dtype=torch.float32)
    state = torch.zeros((8,), dtype=torch.float32)
    noise = torch.zeros((50, 32), dtype=torch.float32)
    tokenized_prompt = torch.tensor([10, 11, 0], dtype=torch.long)
    tokenized_prompt_mask = torch.tensor([True, True, False], dtype=torch.bool)

    runtime.sample_actions_with_timing(
        prompt="client task",
        images=images,
        state=state,
        noise=noise,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    second_actions, second_timing = runtime.sample_actions_with_timing(
        prompt="client task",
        images=images,
        state=state,
        noise=noise,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        executed_steps=1,
    )

    assert torch.all(second_actions == 7.0)
    assert second_timing["accepted_prefix_len"] == 5.0
    assert observed["compiled_full_calls"] == 1
    assert observed["compiled_draft_calls"] == 1
    assert observed["prompt"] == "client task"
    assert observed["full_tokens"] == [10, 11, 0]
    assert observed["full_token_mask"] == [True, True, False]
    assert observed["draft_tokens"] == [10, 11, 0]
    assert observed["draft_token_mask"] == [True, True, False]
    assert observed["encoder_runtime_is_self"] is True
    assert observed["x0_draft_mean"] == 4.0
