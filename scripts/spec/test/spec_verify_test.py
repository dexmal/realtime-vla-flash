import ast
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
import torch

import openpi.models_pytorch.draft as draft_module
import openpi.models_pytorch.spec_pi0_pytorch as spec_pi0_pytorch
import scripts.spec.enc_cache as enc_cache
import scripts.spec.spec_draft_train as spec_draft_train
from openpi.models_pytorch.draft import DraftChunkHead
from openpi.models_pytorch.spec_pi0_pytorch import SpecArgs
from openpi.models_pytorch.spec_pi0_pytorch import SpecPI0Pytorch
from openpi.models_pytorch.spec_pi0_pytorch import _accepted_prefix_len_from_mask
from openpi.models_pytorch.spec_pi0_pytorch import _compute_radius_prefix_acceptance
from openpi.models_pytorch.spec_pi0_pytorch import _full_round_accepted_prefix_len
from openpi.models_pytorch.spec_pi0_pytorch import _populate_full_round_timing
from openpi.models_pytorch.spec_pi0_pytorch import _should_run_full_pipeline_round
from openpi.models_pytorch.spec_pi0_pytorch import _should_schedule_full_fallback
from openpi.models_pytorch.spec_pi0_pytorch import _stitch_radius_prefix_output
from openpi.models_pytorch.spec_pi0_pytorch import _truncate_accepted_prefix_on_gripper_switch
from scripts.spec.spec_draft_train import _compute_draft_training_loss
from scripts.spec.spec_draft_train import _build_draft_head
from scripts.spec.spec_draft_train import _draft_head_meta
from scripts.spec.spec_draft_train import _loss_step_weights
from scripts.spec.spec_draft_train import _require_cache_compatible_with_head
from scripts.spec.spec_draft_train import _run_draft_head
from scripts.spec.spec_draft_train import _ShardCacheDataset
from scripts.spec.spec_draft_train import _weighted_huber_loss
from scripts.spec.spec_draft_train import Args as DraftTrainArgs


def _radius_prefix_stitch_for_test(
    *,
    x0_draft: torch.Tensor,
    x0_hat: torch.Tensor,
    tau_radius: float,
    dist_dims: int,
    eval_h: int,
    gripper_prev: torch.Tensor | None = None,
    enable_gripper_verify: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del gripper_prev, enable_gripper_verify
    accepted_prefix_len, dist = _compute_radius_prefix_acceptance(
        x0_draft=x0_draft,
        x0_hat=x0_hat,
        tau_radius=tau_radius,
        dist_dims=dist_dims,
        eval_h=eval_h,
    )
    x0_out = _stitch_radius_prefix_output(
        x0_draft=x0_draft,
        x0_tail=x0_hat.mean(dim=1),
        accepted_prefix_len=accepted_prefix_len,
    )
    eval_h2 = int(min(int(x0_draft.shape[1]), max(1, int(eval_h))))
    force = torch.zeros((int(x0_draft.shape[0]), eval_h2), device=x0_draft.device, dtype=torch.bool)
    reject = torch.zeros((int(x0_draft.shape[0]), eval_h2), device=x0_draft.device, dtype=torch.bool)
    return x0_out, accepted_prefix_len, dist, force, reject


def test_spec_draft_train_script_help_imports_worktree_openpi() -> None:
    repo_root = str(Path(__file__).resolve().parents[3])
    result = subprocess.run(
        [sys.executable, "scripts/spec/spec_draft_train.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

@pytest.mark.parametrize("module", [enc_cache, spec_draft_train])
def test_setup_ddp_sets_cuda_device_before_init_process_group(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")

    call_order: list[str] = []

    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)

    def _set_device(device: object) -> None:
        call_order.append(f"set_device:{device}")

    def _init_process_group(*, backend: str, init_method: str, device_id: torch.device | None = None) -> None:
        call_order.append(f"init_process_group:{backend}:{init_method}:{device_id}")

    monkeypatch.setattr(module.torch.cuda, "set_device", _set_device)
    monkeypatch.setattr(module.dist, "init_process_group", _init_process_group)

    use_ddp, rank, world_size, local_rank, device = module._setup_ddp()

    assert use_ddp is True
    assert rank == 2
    assert world_size == 4
    assert local_rank == 2
    assert str(device) == "cuda:2"
    assert call_order == [
        "set_device:cuda:2",
        "init_process_group:nccl:env://:cuda:2",
    ]


@pytest.mark.parametrize("module", [enc_cache, spec_draft_train])
def test_dist_barrier_binds_local_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> None:
    called: list[list[int] | None] = []

    def _barrier(*, device_ids: list[int] | None = None, **_: object) -> None:
        called.append(device_ids)

    monkeypatch.setattr(module.dist, "barrier", _barrier)

    module._dist_barrier(torch.device("cuda:3"))
    module._dist_barrier(torch.device("cpu"))

    assert called == [[3], None]


def test_encoder_stage_returns_flat_tensor_tuple_for_torch_compile() -> None:
    class _FakePaliGemma:
        @staticmethod
        def embed_image(img: torch.Tensor) -> torch.Tensor:
            return img

        @staticmethod
        def embed_language_tokens(lang_tokens: torch.Tensor) -> torch.Tensor:
            return lang_tokens

    stub = type("Stub", (), {})()
    stub.paligemma_with_expert = _FakePaliGemma()
    stub._apply_checkpoint = lambda fn, *args: fn(*args)

    images = (
        torch.ones((1, 2, 3), dtype=torch.float32),
        torch.full((1, 2, 3), 2.0, dtype=torch.float32),
    )
    img_masks = (
        torch.ones((1,), dtype=torch.bool),
        torch.ones((1,), dtype=torch.bool),
    )
    lang_tokens = torch.full((1, 1, 3), 3.0, dtype=torch.float32)
    lang_masks = torch.ones((1, 1), dtype=torch.bool)

    outputs = SpecPI0Pytorch._encoder_stage_impl(stub, images, img_masks, lang_tokens, lang_masks)

    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
    assert all(isinstance(x, torch.Tensor) for x in outputs)
    assert outputs[0].shape == (1, 5, 3)
    assert outputs[1].shape == (1, 5)
    assert outputs[2].shape == (1, 5)


def test_radius_prefix_min_k_and_stitch() -> None:
    # B=1, K=2, H=5, D=7
    x0_draft = torch.zeros((1, 5, 7), dtype=torch.float32)
    x0_hat = torch.zeros((1, 2, 5, 7), dtype=torch.float32)

    # k0 accepts 4 steps (0..3), fails at step4.
    x0_hat[0, 0, 4, :] = 2.0
    # k1 accepts 2 steps (0..1), fails at step2+.
    x0_hat[0, 1, 2:, :] = 2.0

    x0_out, accepted, dist, _, _ = _radius_prefix_stitch_for_test(
        x0_draft=x0_draft,
        x0_hat=x0_hat,
        tau_radius=1.0,
        dist_dims=7,
        eval_h=5,
    )

    assert dist.shape == (1, 2, 5)
    assert accepted.shape == (1,)
    assert int(accepted.item()) == 2

    # Prefix uses draft (zeros).
    assert torch.allclose(x0_out[0, 0:2, :], torch.zeros((2, 7)))

    # Suffix uses mean over K.
    # step2: mean(0,2)=1; step3: mean(0,2)=1; step4: mean(2,2)=2
    assert torch.allclose(x0_out[0, 2, :], torch.full((7,), 1.0))
    assert torch.allclose(x0_out[0, 3, :], torch.full((7,), 1.0))
    assert torch.allclose(x0_out[0, 4, :], torch.full((7,), 2.0))


def test_radius_prefix_does_not_average_before_judging() -> None:
    # Construct a case where averaging x0_hat across K would pass, but one k fails.
    x0_draft = torch.zeros((1, 3, 7), dtype=torch.float32)
    x0_hat = torch.zeros((1, 2, 3, 7), dtype=torch.float32)
    x0_hat[0, 0, 1, :] = 2.0  # fail at step1 for k0 (tau=1)
    x0_hat[0, 1, 1, :] = 0.0  # pass at step1 for k1

    x0_out, accepted, _, _, _ = _radius_prefix_stitch_for_test(
        x0_draft=x0_draft,
        x0_hat=x0_hat,
        tau_radius=1.0,
        dist_dims=7,
        eval_h=3,
    )
    assert int(accepted.item()) == 1

    # step0 is draft, step1 is mean tail (=1.0).
    assert torch.allclose(x0_out[0, 0, :], torch.zeros((7,)))
    assert torch.allclose(x0_out[0, 1, :], torch.full((7,), 1.0))


def test_radius_prefix_dist_dims_ignores_tail_dims() -> None:
    # D=10 but we only verify first 7 dims; large diffs in dims 7..9 should not affect acceptance.
    x0_draft = torch.randn((1, 4, 10), dtype=torch.float32)
    x0_hat = x0_draft[:, None, :, :].clone()  # (B,K,H,D)
    x0_hat[0, 0, :, 7:] = 100.0  # huge mismatch outside dist_dims

    x0_out, accepted, _, _, _ = _radius_prefix_stitch_for_test(
        x0_draft=x0_draft,
        x0_hat=x0_hat,
        tau_radius=1e-3,
        dist_dims=7,
        eval_h=4,
    )
    assert int(accepted.item()) == 4
    # Full prefix accepted -> output should be exactly draft.
    assert torch.allclose(x0_out, x0_draft)


def test_radius_prefix_eval_h_limits_verification_window() -> None:
    # Differences beyond eval_h should not reduce accepted_prefix_len.
    x0_draft = torch.zeros((1, 6, 7), dtype=torch.float32)
    x0_hat = torch.zeros((1, 1, 6, 7), dtype=torch.float32)
    x0_hat[0, 0, 3:, :] = 2.0  # would fail at step3 if verified

    x0_out, accepted, _, _, _ = _radius_prefix_stitch_for_test(
        x0_draft=x0_draft,
        x0_hat=x0_hat,
        tau_radius=1.0,
        dist_dims=7,
        eval_h=3,  # only verify steps 0..2
    )
    assert int(accepted.item()) == 3
    # steps 0..2 are from draft; step3 is from tail mean (=2.0)
    assert torch.allclose(x0_out[0, 0:3, :], torch.zeros((3, 7)))
    assert torch.allclose(x0_out[0, 3, :], torch.full((7,), 2.0))


def test_radius_prefix_stitch_keeps_gripper_from_stitched_output() -> None:
    x0_draft = torch.zeros((1, 2, 7), dtype=torch.float32)
    x0_draft[0, :, 6] = 0.4
    x0_hat = x0_draft[:, None, :, :].repeat(1, 2, 1, 1)
    x0_hat[0, :, :, 6] = 0.42

    x0_out, accepted, _, force, reject = _radius_prefix_stitch_for_test(
        x0_draft=x0_draft,
        x0_hat=x0_hat,
        tau_radius=1e-3,
        dist_dims=7,
        eval_h=2,
        gripper_prev=torch.tensor([0.0], dtype=torch.float32),
    )

    assert int(accepted.item()) == 2
    assert not bool(force.any().item())
    assert not bool(reject.any().item())
    assert torch.allclose(x0_out[0, :2, 6], torch.full((2,), 0.4, dtype=torch.float32))


def test_truncate_accepted_prefix_on_gripper_switch_no_switch_in_prefix() -> None:
    x0_out = torch.zeros((1, 4, 7), dtype=torch.float32)
    x0_out[0, :, 6] = torch.tensor([-0.6, -0.5, -0.4, -0.3], dtype=torch.float32)

    accepted, cut = _truncate_accepted_prefix_on_gripper_switch(
        x0_out=x0_out,
        accepted_prefix_len=torch.tensor([3], dtype=torch.int64),
        gripper_prev=torch.tensor([-0.8], dtype=torch.float32),
        gripper_switch_threshold=0.0,
    )

    assert torch.equal(accepted, torch.tensor([3], dtype=torch.int64))
    assert not bool(cut.any().item())


def test_truncate_accepted_prefix_on_gripper_switch_at_first_step() -> None:
    x0_out = torch.zeros((1, 4, 7), dtype=torch.float32)
    x0_out[0, :, 6] = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)

    accepted, cut = _truncate_accepted_prefix_on_gripper_switch(
        x0_out=x0_out,
        accepted_prefix_len=torch.tensor([4], dtype=torch.int64),
        gripper_prev=torch.tensor([-0.1], dtype=torch.float32),
        gripper_switch_threshold=0.0,
    )

    assert torch.equal(accepted, torch.tensor([0], dtype=torch.int64))
    assert torch.equal(cut, torch.tensor([True], dtype=torch.bool))


def test_truncate_accepted_prefix_on_gripper_switch_inside_prefix() -> None:
    x0_out = torch.zeros((1, 5, 7), dtype=torch.float32)
    x0_out[0, :, 6] = torch.tensor([-0.3, -0.2, -0.1, 0.2, 0.3], dtype=torch.float32)

    accepted, cut = _truncate_accepted_prefix_on_gripper_switch(
        x0_out=x0_out,
        accepted_prefix_len=torch.tensor([5], dtype=torch.int64),
        gripper_prev=torch.tensor([-0.4], dtype=torch.float32),
        gripper_switch_threshold=0.0,
    )

    assert torch.equal(accepted, torch.tensor([3], dtype=torch.int64))
    assert torch.equal(cut, torch.tensor([True], dtype=torch.bool))


def test_truncate_accepted_prefix_on_gripper_switch_after_prefix_has_no_effect() -> None:
    x0_out = torch.zeros((1, 5, 7), dtype=torch.float32)
    x0_out[0, :, 6] = torch.tensor([-0.3, -0.2, -0.1, 0.2, 0.3], dtype=torch.float32)

    accepted, cut = _truncate_accepted_prefix_on_gripper_switch(
        x0_out=x0_out,
        accepted_prefix_len=torch.tensor([3], dtype=torch.int64),
        gripper_prev=torch.tensor([-0.4], dtype=torch.float32),
        gripper_switch_threshold=0.0,
    )

    assert torch.equal(accepted, torch.tensor([3], dtype=torch.int64))
    assert not bool(cut.any().item())


def test_truncate_accepted_prefix_on_gripper_switch_uses_x0_out_not_x0_draft() -> None:
    x0_draft = torch.zeros((1, 5, 7), dtype=torch.float32)
    x0_draft[0, :, 6] = torch.tensor([-0.4, -0.3, -0.2, -0.1, -0.1], dtype=torch.float32)
    x0_hat = x0_draft[:, None, :, :].repeat(1, 2, 1, 1)
    x0_hat[0, :, 3:, 6] = 0.6

    x0_out, accepted, _, force, reject = _radius_prefix_stitch_for_test(
        x0_draft=x0_draft,
        x0_hat=x0_hat,
        tau_radius=1.0,
        dist_dims=7,
        eval_h=3,
    )
    truncated, cut = _truncate_accepted_prefix_on_gripper_switch(
        x0_out=x0_out,
        accepted_prefix_len=accepted,
        gripper_prev=torch.tensor([-0.5], dtype=torch.float32),
        gripper_switch_threshold=0.0,
    )

    assert int(accepted.item()) == 3
    assert not bool(force.any().item())
    assert not bool(reject.any().item())
    assert torch.equal(truncated, torch.tensor([3], dtype=torch.int64))
    assert not bool(cut.any().item())


def test_truncate_accepted_prefix_on_gripper_switch_detects_threshold_boundary_crossing() -> None:
    x0_out = torch.zeros((2, 3, 7), dtype=torch.float32)
    x0_out[0, :, 6] = torch.tensor([0.0, -0.1, -0.2], dtype=torch.float32)
    x0_out[1, :, 6] = torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float32)

    accepted, cut = _truncate_accepted_prefix_on_gripper_switch(
        x0_out=x0_out,
        accepted_prefix_len=torch.tensor([3, 3], dtype=torch.int64),
        gripper_prev=torch.tensor([-0.1, 0.1], dtype=torch.float32),
        gripper_switch_threshold=0.0,
    )

    assert torch.equal(accepted, torch.tensor([0, 0], dtype=torch.int64))
    assert torch.equal(cut, torch.tensor([True, True], dtype=torch.bool))


def test_spec_args_defaults_enable_gripper_verify_and_post_verify() -> None:
    args = SpecArgs()
    assert args.enable_gripper_verify is True
    assert getattr(args, "enable_gripper_post_verify", None) is True
    assert not hasattr(args, "gripper_transition_mode")


def test_detect_verify_gripper_switch_any_k_uses_any_k_within_eval_h() -> None:
    detect_fn = getattr(spec_pi0_pytorch, "_detect_verify_gripper_switch_any_k", None)
    assert callable(detect_fn)

    x0_hat = torch.zeros((1, 2, 5, 7), dtype=torch.float32)
    x0_hat[0, 0, :, 6] = torch.tensor([-0.4, -0.3, 0.2, 0.3, 0.4], dtype=torch.float32)
    x0_hat[0, 1, :, 6] = torch.tensor([-0.4, -0.3, -0.2, -0.1, -0.1], dtype=torch.float32)

    triggered = detect_fn(
        x0_hat=x0_hat,
        gripper_prev=torch.tensor([-0.5], dtype=torch.float32),
        gripper_switch_threshold=0.0,
        eval_h=4,
    )

    assert torch.equal(triggered, torch.tensor([True], dtype=torch.bool))


def test_detect_verify_gripper_switch_any_k_ignores_crossing_beyond_eval_h() -> None:
    detect_fn = getattr(spec_pi0_pytorch, "_detect_verify_gripper_switch_any_k", None)
    assert callable(detect_fn)

    x0_hat = torch.zeros((1, 2, 5, 7), dtype=torch.float32)
    x0_hat[0, 0, :, 6] = torch.tensor([-0.4, -0.3, -0.2, -0.1, 0.3], dtype=torch.float32)
    x0_hat[0, 1, :, 6] = torch.tensor([-0.4, -0.3, -0.2, -0.1, -0.1], dtype=torch.float32)

    triggered = detect_fn(
        x0_hat=x0_hat,
        gripper_prev=torch.tensor([-0.5], dtype=torch.float32),
        gripper_switch_threshold=0.0,
        eval_h=4,
    )

    assert torch.equal(triggered, torch.tensor([False], dtype=torch.bool))


def test_accepted_prefix_len_from_mask() -> None:
    mask = torch.tensor(
        [
            [True, True, False, True],
            [False, True, True, True],
            [True, True, True, True],
        ]
    )
    accepted = _accepted_prefix_len_from_mask(mask)
    assert torch.equal(accepted, torch.tensor([2, 0, 4], dtype=torch.int64))


def test_accepted_prefix_len_from_mask_rejects_non_matrix() -> None:
    mask = torch.ones((1, 2, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match="expected accept_mask"):
        _accepted_prefix_len_from_mask(mask)


def test_should_schedule_full_fallback_on_zero_accept_or_gripper_switch_cut() -> None:
    accepted = torch.tensor([0], dtype=torch.int64)
    cut = torch.tensor([False], dtype=torch.bool)
    assert _should_schedule_full_fallback(
        full_fallback=True,
        accepted_prefix_len=accepted,
        gripper_switch_cut_mask=cut,
    ) is True
    assert _should_schedule_full_fallback(
        full_fallback=False,
        accepted_prefix_len=accepted,
        gripper_switch_cut_mask=cut,
    ) is False
    assert _should_schedule_full_fallback(
        full_fallback=True,
        accepted_prefix_len=torch.tensor([2], dtype=torch.int64),
        gripper_switch_cut_mask=cut,
    ) is False
    assert _should_schedule_full_fallback(
        full_fallback=True,
        accepted_prefix_len=torch.tensor([2], dtype=torch.int64),
        gripper_switch_cut_mask=torch.tensor([True], dtype=torch.bool),
    ) is True


def test_spec_args_reject_legacy_gripper_verify_thresholds() -> None:
    with pytest.raises(TypeError):
        SpecArgs(gripper_delta_threshold=0.1)
    with pytest.raises(TypeError):
        SpecArgs(gripper_verify_threshold=0.05)


def test_spec_args_reject_removed_hold_gripper_arg() -> None:
    with pytest.raises(TypeError):
        SpecArgs(hold_gripper=True)

def test_serve_args_accept_force_full_each_round() -> None:
    from scripts.spec.spec_serve_policy import Args as ServeArgs

    args = ServeArgs(force_full_each_round=True)

    assert args.force_full_each_round is True


def test_serve_args_use_named_triton_artifact_paths() -> None:
    from scripts.spec.spec_serve_policy import Args as ServeArgs

    args = ServeArgs(base_triton_path="converted/base", draft_triton_path="converted/draft")

    assert args.base_triton_path == "converted/base"
    assert args.draft_triton_path == "converted/draft"
    assert not hasattr(args, "triton_path")
    assert not hasattr(args, "draft_triton_checkpoint")


def test_triton_runtime_interfaces_do_not_require_realtime_vla_dir() -> None:
    import scripts.spec.triton.triton_pi0_runtime as triton_runtime

    targets = [
        triton_runtime.build_prompt_cache,
        triton_runtime.convert_jax_checkpoint,
        triton_runtime.ensure_triton_checkpoint,
        triton_runtime.create_pi0_inference,
        triton_runtime.create_pi0_spec_inference,
        triton_runtime.load_pi0_inference,
        triton_runtime.TritonRuntimePool.__init__,
        triton_runtime.SpecTritonRuntimePool.__init__,
    ]

    for target in targets:
        assert "realtime_vla_dir" not in inspect.signature(target).parameters


def test_triton_runtime_loads_local_spec_runtime_files(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.spec.triton.triton_pi0_runtime as triton_runtime

    calls: list[tuple[Path, str]] = []

    def _fake_load_module(module_path: Path, module_name: str) -> types.SimpleNamespace:
        calls.append((module_path, module_name))
        return types.SimpleNamespace()

    monkeypatch.setattr(triton_runtime, "_load_module", _fake_load_module)

    triton_runtime._spec_runtime_module("convert_for_triton.py", "convert_probe")
    triton_runtime._spec_runtime_module("pi0_infer.py", "pi0_probe")
    triton_runtime._spec_runtime_module("pi0_spec_infer.py", "spec_probe")

    spec_dir = Path(triton_runtime.__file__).resolve().parent
    assert [(path.resolve(), name) for path, name in calls] == [
        (spec_dir / "convert_for_triton.py", "convert_probe"),
        (spec_dir / "pi0_infer.py", "pi0_probe"),
        (spec_dir / "pi0_spec_infer.py", "spec_probe"),
    ]


def test_triton_serve_args_do_not_expose_realtime_vla_dir() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = repo_root / "scripts" / "spec" / "spec_serve_policy.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    args_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Args")
    field_names = {
        node.target.id
        for node in args_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert "realtime_vla_dir" not in field_names


def test_triton_serve_script_prefers_repo_root_and_src_on_sys_path(tmp_path: Path) -> None:
    import scripts.spec.spec_serve_policy as triton_serve_policy

    repo_root = tmp_path / "some-worktree"
    script_path = repo_root / "scripts" / "spec" / "spec_serve_policy.py"
    src_dir = repo_root / "src"
    script_path.parent.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    script_path.write_text("", encoding="utf-8")

    sys_path = [str(script_path.parent), "/usr/lib/python3/dist-packages"]

    resolved = triton_serve_policy._prefer_local_repo_src(script_path, sys_path)

    assert resolved == src_dir.resolve()
    assert sys_path[:2] == [str(src_dir.resolve()), str(repo_root.resolve())]


def test_triton_sweep_expands_grid_and_uses_libero_client_env() -> None:
    import scripts.spec.exp.run_sweep as run_triton_sweep

    spec = {
        "suites": ["libero_spatial", "libero_goal"],
        "server_base_args": {"port": 8123, "checkpoint_dir": "ckpt"},
        "suite_server_args": {
            "libero_spatial": {
                "cache_dir": "data/triton/pi0_libero_spatial_cache",
                "draft_triton_path": "data/triton/pi0_libero_spatial_cache/draft_triton.pkl",
            },
            "libero_goal": {
                "cache_dir": "data/triton/pi0_libero_goal_cache",
                "draft_triton_path": "data/triton/pi0_libero_goal_cache/draft_triton.pkl",
            },
        },
        "client_base_args": {"num_trials_per_task": 1, "task": "0"},
        "grid": {
            "tau_radius": [0.2, 0.3],
            "t_list": [[0.1, 0.05]],
            "enable_gripper_verify": [True, False],
        },
    }

    runs = run_triton_sweep._iter_run_specs(spec)

    assert len(runs) == 4
    assert runs[0]["server_args"]["checkpoint_dir"] == "ckpt"
    assert runs[0]["server_args"]["port"] == 8123
    assert runs[0]["client_args"]["task"] == "0"
    assert runs[0]["suites"] == ["libero_spatial", "libero_goal"]
    assert len({run["run_id"] for run in runs}) == 4

    server_args = run_triton_sweep._server_args_for_suite(
        runs[0],
        suite="libero_spatial",
        run_dir=Path("out") / "runs" / "run0",
    )
    assert server_args["task_suite_name"] == "libero_spatial"
    assert server_args["cache_dir"] == "data/triton/pi0_libero_spatial_cache"
    assert server_args["draft_triton_path"] == "data/triton/pi0_libero_spatial_cache/draft_triton.pkl"
    assert "--draft-triton-path data/triton/pi0_libero_spatial_cache/draft_triton.pkl" in run_triton_sweep._command_text(
        run_triton_sweep._server_command(server_args, server_python="python")
    )

    client_command = run_triton_sweep._client_command(
        runs[0],
        suite="libero_spatial",
        suite_output_root=Path("out") / "suites",
        server_args=server_args,
        client_python="python",
    )
    shell_command = run_triton_sweep._client_shell_command(
        client_command,
        client_activate="source examples/libero/.venv/bin/activate",
    )
    shell_text = " ".join(shell_command)

    assert shell_command[:2] == ["bash", "-lc"]
    assert "source examples/libero/.venv/bin/activate" in shell_text
    assert "--task-suite-name libero_spatial" in shell_text
    assert "--video-out-path out/suites" in shell_text
    assert "--run-name libero_spatial" in shell_text


def test_triton_sweep_script_prefers_repo_root_on_sys_path(tmp_path: Path) -> None:
    import scripts.spec.exp.run_sweep as run_triton_sweep

    repo_root = tmp_path / "some-worktree"
    script_path = repo_root / "scripts" / "spec" / "exp" / "run_sweep.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("", encoding="utf-8")

    sys_path = [str(script_path.parent), "/usr/lib/python3/dist-packages"]

    resolved = run_triton_sweep._prefer_repo_root(script_path, sys_path)

    assert resolved == repo_root.resolve()
    assert sys_path[:2] == [str(repo_root.resolve()), str(script_path.parent)]


def test_triton_sweep_passes_configured_analyze_warmup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.spec.exp.run_sweep as run_triton_sweep

    spec_path = tmp_path / "sweep.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": "warmup_probe",
                "output_root": str(tmp_path),
                "suites": ["libero_spatial"],
                "client_base_args": {},
                "grid": {"tau_radius": [0.3]},
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[Path, int, bool]] = []

    def _fake_run_one_param_group(**kwargs):
        return {"run_id": kwargs["run_spec"]["run_id"], "status": "completed", "suite_status": {}}

    def _fake_write_summary(*, sweep_dir, warmup_episodes, include_tasks):
        calls.append((Path(sweep_dir), int(warmup_episodes), bool(include_tasks)))
        return {"runs": []}

    monkeypatch.setattr(run_triton_sweep, "_run_one_param_group", _fake_run_one_param_group)
    import scripts.spec.exp.analyze_sweep as analyze_sweep

    monkeypatch.setattr(analyze_sweep, "write_summary", _fake_write_summary)

    sweep_dir = run_triton_sweep.run_sweep(
        run_triton_sweep.Args(
            spec_path=str(spec_path),
            analyze_warmup_episodes=0,
            analyze_include_tasks=True,
            show_progress=False,
        )
    )

    assert len(calls) == 2
    assert calls[0][0].parent == sweep_dir / "runs"
    assert calls[0][1:] == (0, True)
    assert calls[1] == (sweep_dir, 0, True)


def test_analyze_sweep_summarizes_suite_logs_with_sample_latency(tmp_path: Path) -> None:
    import scripts.spec.exp.analyze_sweep as analyze_sweep

    sweep_dir = tmp_path / "sweep"
    run_dir = sweep_dir / "runs" / "tau0.3"
    run_dir.mkdir(parents=True)
    run_spec = {
        "run_id": "tau0.3",
        "params": {"tau_radius": 0.3},
        "server_args": {"tau_radius": 0.3},
        "client_args": {"num_trials_per_task": 2},
        "suites": ["libero_spatial", "libero_goal"],
    }
    (run_dir / "run_spec.json").write_text(json.dumps(run_spec), encoding="utf-8")

    for suite, latency in (("libero_spatial", 10.126), ("libero_goal", 20.567)):
        suite_run_dir = run_dir / "suites" / suite
        suite_run_dir.mkdir(parents=True)
        records = [
            {
                "task_id": 0,
                "task_description": suite,
                "episode_idx": 0,
                "success": True,
                "infer_calls": 2,
                "draft_rounds": 1,
                "vlm_rounds": 0,
                "full_rounds": 1,
                "infer_latency_mean_ms": latency,
                "infer_latency_sum_ms": latency * 2.0,
                "executed_action_count": 4,
                "infer_path": f"episodes/{suite}_success/infer.jsonl",
                "route_ratio_by_action": {"full": 1.0},
                "policy_time_mean_ms": 100.126,
                "serve_time_mean_ms": 110.567,
                "policy_time_sum_ms": 200.252,
                "serve_time_sum_ms": 221.134,
                "accepted_action_len_mean": 0.0,
            },
            {
                "task_id": 1,
                "task_description": f"{suite} failed task",
                "episode_idx": 0,
                "success": False,
                "infer_calls": 3,
                "draft_rounds": 3,
                "vlm_rounds": 0,
                "full_rounds": 0,
                "infer_latency_mean_ms": 999.0,
                "infer_latency_sum_ms": 2997.0,
                "executed_action_count": 3,
                "policy_time_mean_ms": 1000.0,
                "serve_time_mean_ms": 1001.0,
                "policy_time_sum_ms": 3000.0,
                "serve_time_sum_ms": 3003.0,
                "accepted_action_len_mean": 1.0,
            }
        ]
        infer_dir = suite_run_dir / "episodes" / f"{suite}_success"
        infer_dir.mkdir(parents=True)
        (infer_dir / "infer.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"chunk_exec_len": 0, "sample_actions_ms": latency}),
                    json.dumps({"chunk_exec_len": 4, "sample_actions_ms": latency}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (suite_run_dir / "episode_log.json").write_text(json.dumps(records), encoding="utf-8")

    summary = analyze_sweep.write_summary(sweep_dir=sweep_dir, warmup_episodes=0)
    run = summary["runs"][0]
    overall = run["overall"]
    compact_summary = run["summary"]

    assert summary["analysis_scope"] == "sweep"
    assert list(run).index("summary") < list(run).index("overall")
    assert compact_summary == {
        "success_rate_mean": 50.0,
        "avg_latency_per_action_ms": 7.7,
        "infer_latency_mean_ms": 15.3,
        "effective_infer_latency_mean_ms": 30.7,
        "accepted_ratio_mean": 0.0,
        "draft_ratio": 0.0,
        "draft_infer_ratio": 50.0,
        "acc_ratio_by_suite": {"libero_spatial": 0.0, "libero_goal": 0.0},
        "draft_ratio_by_suite": {"libero_spatial": 0.0, "libero_goal": 0.0},
        "success_rate_by_suite": {"libero_spatial": 50.0, "libero_goal": 50.0},
        "avg_latency_per_action_ms_by_suite": {"libero_spatial": 5.1, "libero_goal": 10.3},
        "infer_latency_mean_ms_by_suite": {"libero_spatial": 10.1, "libero_goal": 20.6},
        "effective_infer_latency_mean_ms_by_suite": {"libero_spatial": 20.3, "libero_goal": 41.1},
    }
    assert overall["success_rate_mean"] == 50.0
    assert overall["infer_latency_mean_ms"] == 15.3
    assert overall["effective_infer_latency_mean_ms"] == 30.7
    assert overall["policy_time_mean_ms"] == 100.1
    assert overall["serve_time_mean_ms"] == 110.6
    assert overall["draft_ratio"] == 0.0
    assert overall["full_ratio"] == 100.0
    assert overall["draft_infer_ratio"] == 50.0
    assert overall["full_infer_ratio"] == 50.0
    assert overall["accepted_ratio_mean"] == 0.0
    assert "accepted_action_len_mean" not in overall
    assert overall["acc_ratio_by_suite"] == {"libero_spatial": 0.0, "libero_goal": 0.0}
    assert overall["draft_ratio_by_suite"] == {"libero_spatial": 0.0, "libero_goal": 0.0}
    assert overall["success_rate_by_suite"] == {"libero_spatial": 50.0, "libero_goal": 50.0}
    assert overall["avg_latency_per_action_ms_by_suite"] == {"libero_spatial": 5.1, "libero_goal": 10.3}
    assert overall["infer_latency_mean_ms_by_suite"] == {"libero_spatial": 10.1, "libero_goal": 20.6}
    assert overall["effective_infer_latency_mean_ms_by_suite"] == {"libero_spatial": 20.3, "libero_goal": 41.1}
    assert "tasks" not in summary["runs"][0]["suites"][0]
    assert summary["runs"][0]["suites"][0]["total_tasks"] == 2
    formatted = analyze_sweep._round_run_metrics(
        {
            "server_args": {"max_exec_steps": 4},
            "overall": {
                "accepted_action_len_mean": 2.0,
                "accepted_action_len_mean_by_suite": {"libero_spatial": 2.0},
                "success_rate_mean": 0.25,
                "draft_ratio": 0.5,
                "draft_ratio_by_suite": {"libero_spatial": 0.5},
            },
            "suites": [],
        }
    )
    assert formatted["overall"]["accepted_ratio_mean"] == 50.0
    assert "accepted_action_len_mean" not in formatted["overall"]
    assert formatted["overall"]["acc_ratio_by_suite"] == {"libero_spatial": 50.0}
    assert "accepted_action_len_mean_by_suite" not in formatted["overall"]
    assert formatted["overall"]["draft_ratio_by_suite"] == {"libero_spatial": 50.0}
    assert formatted["overall"]["success_rate_mean"] == 25.0
    assert formatted["overall"]["draft_ratio"] == 50.0

    summary_with_tasks = analyze_sweep.summarize_sweep(
        sweep_dir=sweep_dir,
        warmup_episodes=0,
        include_tasks=True,
    )
    tasks = summary_with_tasks["runs"][0]["suites"][0]["tasks"]
    assert [task["task_id"] for task in tasks] == [0, 1]
    assert tasks[1]["success_rate"] == 0.0
    assert tasks[1]["successful_episodes"] == 0
    assert tasks[1]["latency_episodes_analyzed"] == 0
    assert tasks[1]["latency_episodes_analyzed_all"] == 1
    assert tasks[1]["metric_scope"] == "successful_episodes"
    assert tasks[1]["infer_calls"] is None
    assert tasks[1]["draft_rounds"] is None
    assert tasks[1]["infer_latency_mean_ms"] is None
    assert tasks[1]["policy_time_mean_ms"] is None
    assert tasks[1]["all_episode_metrics"]["infer_calls"] == 3
    assert tasks[1]["all_episode_metrics"]["draft_rounds"] == 3
    assert tasks[1]["all_episode_metrics"]["infer_latency_mean_ms"] == 999.0
    assert tasks[1]["all_episode_metrics"]["policy_time_mean_ms"] == 1000.0
    assert (sweep_dir / "summary.json").exists()
    assert (sweep_dir / "summary.csv").exists()

    run_summary = analyze_sweep.write_summary(sweep_dir=run_dir, warmup_episodes=0)
    assert run_summary["analysis_scope"] == "run"
    assert len(run_summary["runs"]) == 1
    assert run_summary["runs"][0]["run_id"] == "tau0.3"
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.csv").exists()

    legacy_run_dir = tmp_path / "legacy_run"
    legacy_run_dir.mkdir()
    (legacy_run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "legacy_run",
                "task_suite_name": "libero_10",
                "replan_steps": 4,
                "num_trials_per_task": 1,
            }
        ),
        encoding="utf-8",
    )
    legacy_records = [
        {
            "task_id": 0,
            "task_description": "legacy task",
            "episode_idx": 0,
            "success": True,
            "infer_calls": 2,
            "draft_rounds": 1,
            "vlm_rounds": 0,
            "full_rounds": 1,
            "infer_latency_mean_ms": 10.0,
            "infer_latency_sum_ms": 20.0,
            "executed_action_count": 4,
            "infer_path": "episodes/legacy_success/infer.jsonl",
            "route_ratio_by_action": {"draft": 0.5, "full": 0.5},
            "accepted_action_len_mean": 2.0,
        }
    ]
    legacy_infer_dir = legacy_run_dir / "episodes" / "legacy_success"
    legacy_infer_dir.mkdir(parents=True)
    (legacy_infer_dir / "infer.jsonl").write_text(json.dumps({"chunk_exec_len": 4}) + "\n", encoding="utf-8")
    (legacy_run_dir / "episode_log.json").write_text(json.dumps(legacy_records), encoding="utf-8")
    legacy_summary = analyze_sweep.summarize_sweep(sweep_dir=legacy_run_dir)
    legacy_run = legacy_summary["runs"][0]
    assert legacy_summary["analysis_scope"] == "run"
    assert legacy_run["run_id"] == "legacy_run"
    assert legacy_run["suites"][0]["suite"] == "libero_10"
    assert legacy_run["summary"]["accepted_ratio_mean"] == 50.0

    recovered_run_dir = tmp_path / "recovered_run"
    recovered_suite_dir = recovered_run_dir / "suites" / "libero_goal"
    recovered_suite_dir.mkdir(parents=True)
    (recovered_run_dir / "run_spec.json").write_text(
        json.dumps({"run_id": "recovered_run", "suites": ["libero_goal"]}),
        encoding="utf-8",
    )
    (recovered_suite_dir / "episode_log.json").write_text(json.dumps(legacy_records[:1]), encoding="utf-8")
    for task_id in (0, 1):
        episode_dir = recovered_suite_dir / "episodes" / f"task{task_id:02d}_ep000_recovered_task_success"
        episode_dir.mkdir(parents=True)
        (episode_dir / "infer.jsonl").write_text(
            json.dumps(
                {
                    "run_id": "recovered_run",
                    "task_suite_name": "libero_goal",
                    "task_id": task_id,
                    "episode_idx": 0,
                    "route_type": "draft",
                    "accepted_prefix_len": 4,
                    "chunk_exec_len": 4,
                    "sample_actions_ms": 8.0,
                    "policy_time_ms": 9.0,
                    "serve_time_ms": 10.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (episode_dir / "trace.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "task_id": task_id,
                        "episode_idx": 0,
                        "route_type": "draft",
                        "infer_id": 0,
                        "env_step": step,
                        "executed_action": [0.0],
                    }
                )
                + "\n"
                for step in range(4)
            ),
            encoding="utf-8",
        )
    recovered_summary = analyze_sweep.summarize_sweep(sweep_dir=recovered_run_dir)
    recovered_suite = recovered_summary["runs"][0]["suites"][0]
    assert recovered_suite["records_source"] == "episodes"
    assert recovered_suite["total_tasks"] == 2
    assert recovered_suite["total_episodes"] == 2

    duplicate_episode_dir = recovered_suite_dir / "episodes" / "task00_ep000_duplicate_missing_files_success"
    duplicate_episode_dir.mkdir()
    try:
        analyze_sweep.summarize_sweep(sweep_dir=recovered_run_dir)
    except ValueError as exc:
        assert "duplicate episode directories" in str(exc)
        assert "task00_ep000" in str(exc)
    else:
        raise AssertionError("expected duplicate episode directories to raise")

    missing_run_dir = sweep_dir / "runs" / "tau_missing"
    missing_run_dir.mkdir(parents=True)
    (missing_run_dir / "run_spec.json").write_text(
        json.dumps({"run_id": "tau_missing", "params": {"tau_radius": 0.5}, "suites": ["libero_goal"]}),
        encoding="utf-8",
    )
    missing_summary = analyze_sweep.summarize_sweep(sweep_dir=missing_run_dir)
    assert missing_summary["analysis_scope"] == "run"
    assert missing_summary["runs"][0]["suites"][0]["missing"]

    ignored_dir = sweep_dir / "runs" / "tmp"
    ignored_dir.mkdir()
    sweep_summary_after_tmp = analyze_sweep.summarize_sweep(sweep_dir=sweep_dir)
    assert [run["run_id"] for run in sweep_summary_after_tmp["runs"]] == ["tau0.3", "tau_missing"]


def test_analyze_sweep_route_ratios_use_action_weighting_and_old_log_fallback(tmp_path: Path) -> None:
    import scripts.spec.exp.analyze_sweep as analyze_sweep

    records = [
        {
            "task_id": 0,
            "task_description": "short draft task",
            "episode_idx": 0,
            "success": True,
            "infer_calls": 1,
            "draft_rounds": 1,
            "vlm_rounds": 0,
            "full_rounds": 0,
            "executed_action_count": 1,
            "infer_latency_sum_ms": 10.0,
            "infer_path": "episodes/short/infer.jsonl",
            "route_ratio_by_action": {"draft": 1.0},
            "accepted_action_len_mean": 1.0,
        },
        {
            "task_id": 1,
            "task_description": "long full fallback task",
            "episode_idx": 0,
            "success": True,
            "infer_calls": 2,
            "draft_rounds": 1,
            "vlm_rounds": 0,
            "full_rounds": 1,
            "executed_action_count": 9,
            "infer_latency_sum_ms": 900.0,
            "infer_path": "episodes/long/infer.jsonl",
            "route_ratio_by_action": {"full": 1.0},
            "accepted_action_len_mean": 0.0,
        },
    ]
    (tmp_path / "episodes" / "short").mkdir(parents=True)
    (tmp_path / "episodes" / "short" / "infer.jsonl").write_text(
        json.dumps({"chunk_exec_len": 1}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "episodes" / "long").mkdir(parents=True)
    (tmp_path / "episodes" / "long" / "infer.jsonl").write_text(
        "".join(json.dumps({"chunk_exec_len": 1}) + "\n" for _ in range(9)),
        encoding="utf-8",
    )

    summary = analyze_sweep._summarize_episode_records(
        records,
        warmup_episodes=0,
        include_tasks=True,
        episode_log_dir=tmp_path,
    )

    assert summary["draft_ratio"] == pytest.approx(0.1)
    assert summary["full_ratio"] == pytest.approx(0.9)
    assert summary["draft_infer_ratio"] == pytest.approx(0.75)
    assert summary["effective_infer_latency_mean_ms"] == pytest.approx(91.0)
    assert summary["avg_latency_per_action_ms"] == pytest.approx(91.0)
    assert summary["tasks"][1]["draft_ratio"] == 0.0
    assert summary["tasks"][1]["draft_infer_ratio"] == 0.5

    old_log_records = [
        {
            "task_id": 0,
            "task_description": "old task",
            "episode_idx": 0,
            "success": True,
            "infer_calls": 2,
            "draft_rounds": 1,
            "vlm_rounds": 0,
            "full_rounds": 1,
            "executed_action_count": 9,
            "accepted_action_len_mean": 0.0,
        }
    ]
    old_summary = analyze_sweep._summarize_episode_records(old_log_records, warmup_episodes=0, include_tasks=True)

    assert old_summary["draft_ratio"] == old_summary["draft_infer_ratio"] == 0.5
    assert old_summary["full_ratio"] == old_summary["full_infer_ratio"] == 0.5

    run_dir = tmp_path / "mixed_run"
    (run_dir / "suites" / "new").mkdir(parents=True)
    (run_dir / "suites" / "old").mkdir(parents=True)
    (run_dir / "run_spec.json").write_text(
        json.dumps({"run_id": "mixed_run", "suites": ["new", "old"]}),
        encoding="utf-8",
    )
    (run_dir / "suites" / "new" / "episode_log.json").write_text(json.dumps(records[1:]), encoding="utf-8")
    (run_dir / "suites" / "old" / "episode_log.json").write_text(json.dumps(old_log_records), encoding="utf-8")

    run_summary = analyze_sweep.summarize_sweep(sweep_dir=run_dir)
    run_overall = run_summary["runs"][0]["overall"]

    assert run_overall["draft_ratio"] == run_overall["draft_infer_ratio"] == 50.0
    assert run_overall["full_ratio"] == run_overall["full_infer_ratio"] == 50.0


def test_serve_script_prefers_repo_local_src_on_sys_path(tmp_path) -> None:
    import scripts.spec.spec_serve_policy as spec_serve_policy

    repo_root = tmp_path / "some-worktree"
    script_path = repo_root / "scripts" / "spec" / "spec_serve_policy.py"
    src_dir = repo_root / "src"
    script_path.parent.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    script_path.write_text("", encoding="utf-8")

    sys_path = ["/root/code/openpi/src", str(src_dir), "/usr/lib/python3/dist-packages"]

    resolved = spec_serve_policy._prefer_local_repo_src(script_path, sys_path)

    assert resolved == src_dir.resolve()
    assert sys_path[0] == str(src_dir.resolve())
    assert sys_path.count(str(src_dir.resolve())) == 1


def test_serve_route_type_no_longer_exposes_vlm_round() -> None:
    import scripts.spec.spec_serve_policy as spec_serve_policy

    assert spec_serve_policy.TritonServerPolicy._route_type_from_timing({"is_full_pipeline_round": 1.0}) == "full"
    assert spec_serve_policy.TritonServerPolicy._route_type_from_timing({"did_prefill": 1.0}) == "draft"


def test_client_route_type_no_longer_exposes_vlm_round(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("scripts.spec.spec_client_libero", None)
    monkeypatch.setitem(sys.modules, "libero", types.ModuleType("libero"))
    libero_libero = types.ModuleType("libero.libero")
    libero_libero.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {})
    libero_libero.get_libero_path = lambda *_args, **_kwargs: ""
    monkeypatch.setitem(sys.modules, "libero.libero", libero_libero)
    libero_envs = types.ModuleType("libero.libero.envs")
    libero_envs.OffScreenRenderEnv = object
    monkeypatch.setitem(sys.modules, "libero.libero.envs", libero_envs)

    spec_client_libero = importlib.import_module("scripts.spec.spec_client_libero")

    assert spec_client_libero._route_type_from_timing({"is_full_pipeline_round": 1.0}) == "full"
    assert spec_client_libero._route_type_from_timing({"did_prefill": 1.0}) == "draft"
    sys.modules.pop("scripts.spec.spec_client_libero", None)


def test_client_infer_record_adds_policy_and_serve_time_without_changing_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("scripts.spec.spec_client_libero", None)
    monkeypatch.setitem(sys.modules, "libero", types.ModuleType("libero"))
    libero_libero = types.ModuleType("libero.libero")
    libero_libero.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {})
    libero_libero.get_libero_path = lambda *_args, **_kwargs: ""
    monkeypatch.setitem(sys.modules, "libero.libero", libero_libero)
    libero_envs = types.ModuleType("libero.libero.envs")
    libero_envs.OffScreenRenderEnv = object
    monkeypatch.setitem(sys.modules, "libero.libero.envs", libero_envs)

    spec_client_libero = importlib.import_module("scripts.spec.spec_client_libero")

    record = spec_client_libero._make_infer_record(
        run_id="run",
        task_suite_name="libero_goal",
        task_id=1,
        episode_idx=2,
        infer_id=3,
        frame_idx=4,
        env_step=5,
        route_type="draft",
        accepted_prefix_len=6,
        chunk_exec_len=6,
        policy_timing={
            "sample_actions_ms": 11.0,
            "total_ms": 7.0,
            "did_prefill": 0.0,
            "is_full_pipeline_round": 0.0,
            "used_full_fallback": 0.0,
            "scheduled_full_fallback": 0.0,
            "verify_mode_random": 0.0,
            "gripper_verify_enabled": 1.0,
        },
        server_timing={
            "policy_time_ms": 21.0,
            "serve_time_ms": 24.0,
            "ws_unpack_ms": 1.0,
            "ws_pack_ms": 2.0,
            "server_recv_timestamp_s": 100.0,
            "server_response_timestamp_s": 101.0,
        },
        client_send_timestamp_s=90.0,
        client_recv_timestamp_s=91.0,
        client_roundtrip_ms=1000.0,
        chunk_actions=np.zeros((2, 7), dtype=np.float32),
    )

    assert record["sample_actions_ms"] == 11.0
    assert record["policy_time_ms"] == 21.0
    assert record["serve_time_ms"] == 24.0
    assert record["timestamp"] == 91.0
    assert record["client_roundtrip_ms"] == 1000.0
    sys.modules.pop("scripts.spec.spec_client_libero", None)


def test_runtime_module_exports_only_single_head_helpers() -> None:
    assert not hasattr(spec_pi0_pytorch, "_accumulate_refresh_score")
    assert not hasattr(spec_pi0_pytorch, "_compute_refresh_delta")
    assert not hasattr(spec_pi0_pytorch, "_summarize_refresh_tokens")
    assert not hasattr(spec_pi0_pytorch, "_radius_prefix_stitch")


def test_sampling_entrypoints_delegate_to_shared_runtime_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int | None, bool]] = []
    expected_actions = torch.ones((1, 2, 7), dtype=torch.float32)

    def _fake_impl(self, device, observation, noise=None, num_steps=10, executed_steps=None, collect_timing=False):
        del self, device, observation, noise
        calls.append((int(num_steps), executed_steps, bool(collect_timing)))
        if collect_timing:
            return expected_actions.clone(), {"total_ms": 1.0}
        return expected_actions.clone()

    monkeypatch.setattr(SpecPI0Pytorch, "_sample_actions_impl", _fake_impl, raising=False)
    stub = object.__new__(SpecPI0Pytorch)
    observation = type("Obs", (), {"state": torch.zeros((1, 32), dtype=torch.float32)})()
    noise = torch.zeros((1, 2, 7), dtype=torch.float32)

    actions = SpecPI0Pytorch.sample_actions(stub, "cpu", observation, noise=noise, num_steps=5, executed_steps=2)
    actions_timed, timing = SpecPI0Pytorch.sample_actions_with_timing(
        stub,
        "cpu",
        observation,
        noise=noise,
        num_steps=6,
        executed_steps=3,
    )

    assert torch.equal(actions, expected_actions)
    assert torch.equal(actions_timed, expected_actions)
    assert timing == {"total_ms": 1.0}
    assert calls == [(5, 2, False), (6, 3, True)]

def test_compute_draft_requires_loaded_head() -> None:
    stub = type("Stub", (), {})()
    stub._draft_head = None
    stub.spec_args = type("SpecArgsStub", (), {})()

    with pytest.raises(ValueError, match="learned draft head"):
        SpecPI0Pytorch._compute_draft(
            stub,
            noise=torch.zeros((1, 4, 7), dtype=torch.float32),
            prefix_embs=torch.zeros((1, 5, 8), dtype=torch.float32),
            prefix_pad_masks=torch.ones((1, 5), dtype=torch.bool),
            prefix_att_masks=torch.zeros((1, 5), dtype=torch.bool),
            robot_state=torch.zeros((1, 32), dtype=torch.float32),
            last_actions=torch.zeros((1, 6, 7), dtype=torch.float32),
        )

def test_make_draft_head_builds_current_head_only() -> None:
    stub = type("Stub", (), {})()
    stub.spec_args = SpecArgs()
    stub._chunk_m = 4
    stub._draft_history_len = 6

    head = SpecPI0Pytorch._make_draft_head(
        stub,
        img_dim=8,
        device=torch.device("cpu"),
    )
    assert isinstance(head, DraftChunkHead)


def test_make_draft_head_matches_runtime_dtype_and_uses_sdpa_attention() -> None:
    class _RuntimeStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
            self.spec_args = SpecArgs()
            self._chunk_m = 4

    head = SpecPI0Pytorch._make_draft_head(
        _RuntimeStub(),
        img_dim=8,
        device=torch.device("cpu"),
    )

    assert head._gemma_block.self_attn.q_proj.weight.dtype == torch.bfloat16
    assert head._gemma_block.self_attn.config._attn_implementation == "sdpa"


def test_compute_draft_prefers_bound_predictor_and_preserves_prefix_dtype() -> None:
    calls: list[torch.dtype] = []

    class _HeadStub:
        def forward(self, **_kwargs) -> torch.Tensor:
            raise AssertionError("expected _compute_draft to use the bound draft predictor")

    def _bound_predictor(**kwargs) -> torch.Tensor:
        calls.append(kwargs["prefix_embs"].dtype)
        return torch.zeros((1, 4, 7), dtype=torch.float32)

    stub = type("Stub", (), {})()
    stub._draft_head = _HeadStub()
    stub._draft_predict_actions = _bound_predictor
    stub._chunk_m = 4

    x0 = SpecPI0Pytorch._compute_draft(
        stub,
        noise=torch.zeros((1, 6, 7), dtype=torch.float32),
        prefix_embs=torch.randn((1, 5, 8), dtype=torch.bfloat16),
        prefix_pad_masks=torch.ones((1, 5), dtype=torch.bool),
        prefix_att_masks=torch.zeros((1, 5), dtype=torch.bool),
        robot_state=torch.zeros((1, 32), dtype=torch.float32),
        last_actions=torch.randn((1, 6, 7), dtype=torch.float32),
    )

    assert calls == [torch.bfloat16]
    assert x0.shape == (1, 6, 7)


def test_init_spec_modules_compiles_draft_predictor_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    compile_calls: list[str] = []

    class _VisionConfig:
        projection_dim = 8

    class _PaliGemmaConfig:
        vision_config = _VisionConfig()

    class _PaliGemma:
        config = _PaliGemmaConfig()

    class _WithExpert:
        paligemma = _PaliGemma()

    def _fake_compile(fn, *, mode):
        compile_calls.append(mode)
        return fn

    monkeypatch.setenv("OPENPI_COMPILE_DRAFT", "1")
    monkeypatch.delenv("OPENPI_DISABLE_TORCH_COMPILE", raising=False)
    monkeypatch.setattr(spec_pi0_pytorch.torch, "compile", _fake_compile)

    runtime = object.__new__(SpecPI0Pytorch)
    torch.nn.Module.__init__(runtime)
    runtime._anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
    runtime._chunk_m = 4
    runtime.spec_args = SpecArgs()
    runtime._draft_head = None
    runtime._draft_predict_actions = None
    runtime.paligemma_with_expert = _WithExpert()

    SpecPI0Pytorch.init_spec_modules(runtime)

    assert runtime._draft_head is not None
    assert callable(runtime._draft_predict_actions)
    assert compile_calls == ["max-autotune"]


def test_spec_args_reject_oracle_kwargs_and_runtime_has_no_oracle_api() -> None:
    with pytest.raises(TypeError):
        SpecArgs(oracle_force_prefill_each_call=True)
    with pytest.raises(TypeError):
        SpecArgs(oracle_refresh_every_calls=1)
    assert not hasattr(SpecPI0Pytorch, "sample_actions_with_timing_oracle")


def test_training_module_exports_only_single_head_helpers() -> None:
    assert not hasattr(spec_draft_train, "_first_crossing_mask")
    assert not hasattr(spec_draft_train, "_transition_step_weights")
    assert not hasattr(spec_draft_train, "_transition_step_mask")


def test_spec_heads_exports_only_current_draft_head() -> None:
    assert hasattr(draft_module, "DraftChunkHead")
    assert not hasattr(draft_module, "DraftChunkHeadV1")
    assert not hasattr(draft_module, "DraftChunkHeadV15")


def test_training_build_draft_head_returns_current_head_only() -> None:
    head = _build_draft_head(
        img_dim=8,
        chunk_m=4,
        out_dim=7,
        device="cpu",
    )
    assert isinstance(head, DraftChunkHead)


def test_training_build_draft_head_infers_hidden_dim_from_cache_gemma_block_state_dict() -> None:
    head = _build_draft_head(
        img_dim=8,
        chunk_m=4,
        out_dim=7,
        device="cpu",
        state_dict={"mlp.gate_proj.weight": torch.empty((16384, 1), dtype=torch.float32)},
    )

    assert isinstance(head, DraftChunkHead)
    assert head._gemma_block.mlp.gate_proj.weight.shape[0] == 16384


def test_training_prefers_resume_state_dict_for_build_and_falls_back_to_cache_init() -> None:
    cache_state = {"mlp.gate_proj.weight": torch.empty((16384, 1), dtype=torch.float32)}
    resume_state = {"_gemma_block.mlp.gate_proj.weight": torch.empty((8192, 1), dtype=torch.float32)}

    selected_cache = spec_draft_train._select_build_state_dict_for_draft_head({}, ({}, cache_state))
    selected_resume = spec_draft_train._select_build_state_dict_for_draft_head(resume_state, ({}, cache_state))

    assert selected_cache is cache_state
    assert selected_resume is resume_state


def test_training_draft_head_meta_reports_single_current_head() -> None:
    head = DraftChunkHead(
        img_dim=8,
        chunk_m=4,
        hidden_dim=16,
        out_dim=7,
    )

    meta = _draft_head_meta(head)

    assert "draft_hidden_dim" not in meta
    assert "draft_gripper_mode" not in meta
    assert meta["draft_arch"] == "vlm_block"
    assert meta["draft_input_mode"] == "prefix_embs"


def test_should_run_full_pipeline_round_uses_cache_miss_or_full_fallback() -> None:
    assert _should_run_full_pipeline_round(
        cache_ready=False,
        full_fallback=False,
        pending_full_fallback=False,
    ) is True
    assert _should_run_full_pipeline_round(
        cache_ready=True,
        full_fallback=True,
        pending_full_fallback=True,
    ) is True
    assert _should_run_full_pipeline_round(
        cache_ready=True,
        full_fallback=True,
        pending_full_fallback=False,
    ) is False


def test_should_run_full_pipeline_round_honors_force_full_each_round() -> None:
    assert _should_run_full_pipeline_round(
        cache_ready=True,
        full_fallback=False,
        pending_full_fallback=False,
        force_full_each_round=True,
    ) is True


def test_should_run_full_pipeline_round_honors_periodic_full_cadence() -> None:
    assert _should_run_full_pipeline_round(
        cache_ready=True,
        full_fallback=False,
        pending_full_fallback=False,
        periodic_full_every_n_draft_rounds=3,
        draft_rounds_since_full=2,
    ) is False
    assert _should_run_full_pipeline_round(
        cache_ready=True,
        full_fallback=False,
        pending_full_fallback=False,
        periodic_full_every_n_draft_rounds=3,
        draft_rounds_since_full=3,
    ) is True
    assert _should_run_full_pipeline_round(
        cache_ready=True,
        full_fallback=False,
        pending_full_fallback=False,
        periodic_full_every_n_draft_rounds=0,
        draft_rounds_since_full=99,
    ) is False


def test_spec_args_defaults_gripper_full_window_to_one() -> None:
    args = SpecArgs()
    assert args.gripper_full_window == 1

def test_spec_args_defaults_force_full_each_round_to_false() -> None:
    args = SpecArgs()
    assert args.force_full_each_round is False


def test_spec_args_defaults_periodic_full_every_n_draft_rounds_to_zero() -> None:
    args = SpecArgs()
    assert args.periodic_full_every_n_draft_rounds == 0


def test_reset_runtime_state_clears_cache_without_refresh_flags() -> None:
    stub = type("Stub", (), {})()
    stub._score_total = torch.ones((1,), dtype=torch.float32)
    stub._last_actions = torch.ones((1, 6, 7), dtype=torch.float32)
    stub._action_chunk_cache = torch.ones((1, 4, 7), dtype=torch.float32)
    stub._action_cache_ptr = 2
    stub._draft_rounds_since_full = 2
    stub._pending_full_fallback = True
    stub._gripper_full_rounds_left = 1
    stub._past_key_values_cache = object()
    stub._past_kv_prefix_len = 12
    stub._past_kv_batch_size = 1

    SpecPI0Pytorch.reset_runtime_state(stub, force_prefill=True)

    assert stub._pending_full_fallback is False
    assert stub._gripper_full_rounds_left == 0
    assert stub._draft_rounds_since_full == 0
    assert stub._past_key_values_cache is None
    assert stub._past_kv_prefix_len is None
    assert stub._past_kv_batch_size is None
    assert not hasattr(stub, "_score_total")
    assert not hasattr(stub, "_need_refresh_next")
    assert not hasattr(stub, "_pending_initial_full_round")
    assert not hasattr(stub, "_prev_refresh_front")
    assert not hasattr(stub, "_prev_refresh_wrist")


def test_schedule_gripper_full_fallback_uses_window_length() -> None:
    stub = type("Stub", (), {})()
    stub.spec_args = SpecArgs(gripper_full_window=2)
    stub._pending_full_fallback = False
    stub._gripper_full_rounds_left = 0

    SpecPI0Pytorch._schedule_gripper_full_fallback(stub)

    assert stub._pending_full_fallback is True
    assert stub._gripper_full_rounds_left == 2
    assert not hasattr(stub, "_need_refresh_next")


def test_accept_full_round_actions_keeps_gripper_pending_until_window_expires_and_updates_cache() -> None:
    stub = type("Stub", (), {})()
    stub._pending_full_fallback = True
    stub._gripper_full_rounds_left = 2
    stub._draft_rounds_since_full = 3
    stub._action_chunk_cache = None
    stub._action_cache_ptr = 7
    stub._past_key_values_cache = None
    stub._past_kv_prefix_len = None
    stub._past_kv_batch_size = None

    actions = torch.ones((1, 4, 7), dtype=torch.float32)
    past_key_values = [
        (
            torch.full((1, 1, 1, 1), 3.0, dtype=torch.float32),
            torch.full((1, 1, 1, 1), 3.0, dtype=torch.float32),
        )
    ]
    SpecPI0Pytorch._accept_full_round_actions(
        stub,
        actions,
        past_key_values=past_key_values,
        prefix_len=12,
        batch_size=1,
    )

    assert stub._pending_full_fallback is True
    assert stub._gripper_full_rounds_left == 1
    assert stub._draft_rounds_since_full == 0
    assert stub._action_cache_ptr == 0
    assert stub._past_key_values_cache is past_key_values
    assert stub._past_kv_prefix_len == 12
    assert stub._past_kv_batch_size == 1

    SpecPI0Pytorch._accept_full_round_actions(
        stub,
        actions,
        past_key_values=past_key_values,
        prefix_len=12,
        batch_size=1,
    )

    assert stub._pending_full_fallback is False
    assert stub._gripper_full_rounds_left == 0


def test_accept_full_round_actions_clears_non_gripper_pending_fallback_immediately() -> None:
    stub = type("Stub", (), {})()
    stub._pending_full_fallback = True
    stub._gripper_full_rounds_left = 0
    stub._draft_rounds_since_full = 5
    stub._action_chunk_cache = None
    stub._action_cache_ptr = 3
    stub._past_key_values_cache = None
    stub._past_kv_prefix_len = None
    stub._past_kv_batch_size = None

    SpecPI0Pytorch._accept_full_round_actions(
        stub,
        torch.ones((1, 2, 7), dtype=torch.float32),
        past_key_values=[(torch.ones((1, 1, 1, 1), dtype=torch.float32), torch.ones((1, 1, 1, 1), dtype=torch.float32))],
        prefix_len=9,
        batch_size=1,
    )

    assert stub._pending_full_fallback is False
    assert stub._gripper_full_rounds_left == 0
    assert stub._draft_rounds_since_full == 0
    assert stub._past_kv_prefix_len == 9
    assert stub._past_kv_batch_size == 1


def test_full_round_accepted_prefix_len_uses_exec_window() -> None:
    assert _full_round_accepted_prefix_len(action_horizon=50, max_exec_steps=12) == 12
    assert _full_round_accepted_prefix_len(action_horizon=3, max_exec_steps=12) == 3
    assert _full_round_accepted_prefix_len(action_horizon=10, max_exec_steps=0) == 1


def test_populate_full_round_timing_keeps_prefill_separate() -> None:
    timing = {"encoder_ms": 11.0}
    accepted = _populate_full_round_timing(
        timing,
        verify_mode="radius",
        action_horizon=50,
        max_exec_steps=12,
        full_prefill_ms=24.0,
        full_action_ms=60.0,
        gripper_verify_enabled=True,
    )

    assert accepted == 12.0
    assert timing["vlm_prefill_ms"] == 24.0
    assert timing["draft_ms"] == 0.0
    assert timing["action_verify_ms"] == 0.0
    assert timing["full_fallback_ms"] == 60.0
    assert timing["total_ms"] == 95.0
    assert timing["used_full_fallback"] == 1.0
    assert timing["include_in_draft_accept_metrics"] == 0.0
    assert timing["gripper_verify_enabled"] == 1.0


def test_make_speculative_metrics_keeps_only_live_values() -> None:
    make_metrics = getattr(spec_pi0_pytorch, "_make_speculative_metrics", None)
    assert callable(make_metrics)

    metrics = make_metrics(
        radius_dist=torch.tensor(1.5, dtype=torch.float32),
        accepted_prefix_len_mean=torch.tensor(4.0, dtype=torch.float32),
        gripper_switch_cut_rate=torch.tensor(0.25, dtype=torch.float32),
        scheduled_full_fallback_gripper=torch.tensor(1.0, dtype=torch.float32),
        gripper_verify_stop_rate=torch.tensor(0.5, dtype=torch.float32),
    )

    assert tuple(metrics.shape) == (5,)
    assert torch.allclose(metrics, torch.tensor([1.5, 4.0, 0.25, 1.0, 0.5], dtype=torch.float32))


def test_set_legacy_timing_compat_fields_populates_removed_refresh_keys() -> None:
    compat_fn = getattr(spec_pi0_pytorch, "_set_legacy_timing_compat_fields", None)
    assert callable(compat_fn)

    timing = {"encoder_ms": 11.0}
    compat_fn(timing)

    assert timing["enc_priority_mean"] == 0.0
    assert timing["score_delta"] == 0.0
    assert timing["score_total"] == 0.0
    assert timing["suggest_refresh"] == 0.0
    assert timing["gripper_override_rate"] == 0.0
    assert timing["gripper_force_rate"] == 0.0
    assert timing["gripper_reject_rate"] == 0.0
    assert timing["encoder_ms"] == 11.0


def test_get_cached_past_key_values_returns_none_on_cache_miss_or_shape_change() -> None:
    prefix_pad_masks = torch.ones((1, 2), dtype=torch.bool)
    stub = type("Stub", (), {})()
    stub._past_key_values_cache = None
    stub._past_kv_prefix_len = None
    stub._past_kv_batch_size = None

    assert SpecPI0Pytorch._get_cached_past_key_values(stub, prefix_pad_masks) is None

    cached = [(torch.full((1, 1, 1, 1), 7.0, dtype=torch.float32), torch.full((1, 1, 1, 1), 7.0, dtype=torch.float32))]
    stub._past_key_values_cache = cached
    stub._past_kv_prefix_len = 3
    stub._past_kv_batch_size = 1

    assert SpecPI0Pytorch._get_cached_past_key_values(stub, prefix_pad_masks) is None


def test_get_cached_past_key_values_reuses_matching_cache() -> None:
    prefix_pad_masks = torch.ones((1, 2), dtype=torch.bool)
    cached = [(torch.full((1, 1, 1, 1), 7.0, dtype=torch.float32), torch.full((1, 1, 1, 1), 7.0, dtype=torch.float32))]

    stub = type("Stub", (), {})()
    stub._past_key_values_cache = cached
    stub._past_kv_prefix_len = 2
    stub._past_kv_batch_size = 1

    cache = SpecPI0Pytorch._get_cached_past_key_values(stub, prefix_pad_masks)

    assert cache is cached


def test_draft_train_defaults_keep_current_vlm_block_loss_knobs_without_legacy_bonus_knobs() -> None:
    args = DraftTrainArgs()

    assert args.loss_prefix_mode == "sampled_prefix"
    assert args.loss_prefix_cap == 16
    assert not hasattr(args, "transition_pre_steps")
    assert not hasattr(args, "transition_post_steps")

def test_vlm_block_draft_training_path_returns_action_only_and_finite_loss() -> None:
    head = DraftChunkHead(
        img_dim=8,
        chunk_m=4,
        hidden_dim=16,
        out_dim=7,
    )
    prefix_embs = torch.randn((2, 16, 8), dtype=torch.float32)
    prefix_pad_masks = torch.ones((2, 16), dtype=torch.bool)
    prefix_att_masks = torch.zeros((2, 16), dtype=torch.bool)
    robot_state = torch.randn((2, 32), dtype=torch.float32)
    last_actions = torch.randn((2, 6, 7), dtype=torch.float32)
    pred_actions, gripper_aux = _run_draft_head(
        head,
        primary_input=prefix_embs,
        secondary_input=prefix_pad_masks,
        tertiary_input=prefix_att_masks,
        robot_state=robot_state,
        last_actions=last_actions,
    )
    target_actions = torch.randn_like(pred_actions)
    loss = _compute_draft_training_loss(
        pred_actions=pred_actions,
        target_actions=target_actions,
        gripper_aux=gripper_aux,
        gamma=0.95,
        loss_prefix_mode="full_chunk_gamma",
        gamma_prefix=0.9,
        gamma_tail=1.0,
        tail_weight=0.1,
        beta=1.0,
        gripper_loss_weight=1.0,
        gripper_delta_weight=0.5,
        gripper_transition_threshold=0.1,
        gripper_transition_weight=3.0,
        prefix_cap=4,
        device=pred_actions.device,
    )

    assert gripper_aux is None
    assert pred_actions.shape == (2, 4, 7)
    assert torch.isfinite(loss)


def test_vlm_block_draft_head_accepts_prefix_embeddings_and_ignores_last_actions() -> None:
    head = DraftChunkHead(
        img_dim=8,
        chunk_m=4,
        hidden_dim=16,
        out_dim=7,
    )
    prefix_embs = torch.randn((2, 5, 8), dtype=torch.float32)
    prefix_pad_masks = torch.ones((2, 5), dtype=torch.bool)
    prefix_att_masks = torch.zeros((2, 5), dtype=torch.bool)
    robot_state = torch.randn((2, 32), dtype=torch.float32)
    last_actions_a = torch.randn((2, 6, 7), dtype=torch.float32)
    last_actions_b = torch.randn((2, 6, 7), dtype=torch.float32)

    outputs_a = head(
        prefix_embs=prefix_embs,
        prefix_pad_masks=prefix_pad_masks,
        prefix_att_masks=prefix_att_masks,
        robot_state=robot_state,
        last_actions=last_actions_a,
    )
    outputs_b = head(
        prefix_embs=prefix_embs,
        prefix_pad_masks=prefix_pad_masks,
        prefix_att_masks=prefix_att_masks,
        robot_state=robot_state,
        last_actions=last_actions_b,
    )

    assert outputs_a.shape == (2, 4, 7)
    assert torch.allclose(outputs_a, outputs_b)


def test_vlm_block_draft_head_builds_one_way_full_query_attention_mask() -> None:
    head = DraftChunkHead(
        img_dim=8,
        chunk_m=2,
        hidden_dim=16,
        out_dim=7,
    )
    prefix_pad_masks = torch.ones((1, 3), dtype=torch.bool)

    mask = head._build_attention_mask(prefix_pad_masks=prefix_pad_masks)

    assert mask.shape == (1, 6, 6)
    # Prefix tokens cannot attend query tokens.
    assert torch.equal(mask[0, 0, 4:], torch.zeros((2,), dtype=torch.bool))
    # Queries can attend prefix and each other.
    assert torch.equal(mask[0, 4, :], torch.ones((6,), dtype=torch.bool))
    assert torch.equal(mask[0, 5, :], torch.ones((6,), dtype=torch.bool))


def test_compute_draft_vlm_block_uses_prefix_embeddings_for_runtime() -> None:
    stub = type("Stub", (), {})()
    stub._draft_head = DraftChunkHead(
        img_dim=8,
        chunk_m=4,
        hidden_dim=16,
        out_dim=7,
    )
    stub._chunk_m = 4

    x0 = SpecPI0Pytorch._compute_draft(
        stub,
        noise=torch.zeros((1, 6, 7), dtype=torch.float32),
        prefix_embs=torch.randn((1, 5, 8), dtype=torch.float32),
        prefix_pad_masks=torch.ones((1, 5), dtype=torch.bool),
        prefix_att_masks=torch.zeros((1, 5), dtype=torch.bool),
        robot_state=torch.zeros((1, 32), dtype=torch.float32),
        last_actions=torch.randn((1, 6, 7), dtype=torch.float32),
    )

    assert x0.shape == (1, 6, 7)


def test_training_draft_head_meta_reports_vlm_block_without_history_or_seed() -> None:
    head = DraftChunkHead(
        img_dim=8,
        chunk_m=4,
        hidden_dim=16,
        out_dim=7,
    )

    meta = _draft_head_meta(head)

    assert meta["draft_arch"] == "vlm_block"
    assert meta["draft_input_mode"] == "prefix_embs"
    assert "draft_hidden_dim" not in meta
    assert meta["use_last_actions"] is False
    assert meta["use_seed_actions"] is False


def test_shard_cache_dataset_reads_prefix_embeddings_when_manifest_requests_prefix_mode(tmp_path) -> None:
    run_dir = tmp_path / "cache"
    run_dir.mkdir()
    save_safetensors(
        {
            "prefix_embs": torch.randn((1, 16, 8), dtype=torch.float32),
            "prefix_pad_masks": torch.ones((1, 16), dtype=torch.bool),
            "prefix_att_masks": torch.zeros((1, 16), dtype=torch.bool),
            "robot_state": torch.randn((1, 32), dtype=torch.float32),
            "last_actions": torch.randn((1, 6, 7), dtype=torch.float32),
            "seed_actions": torch.randn((1, 2, 7), dtype=torch.float32),
            "targets": torch.randn((1, 4, 7), dtype=torch.float32),
        },
        str(run_dir / "rank000_shard00000.safetensors"),
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "target_source": "teacher_zero_noise",
                "teacher_noise_mode": "zero",
                "sample_semantics": "sliding_chunk_shift",
                "draft_arch": "vlm_block",
                "draft_input_mode": "prefix_embs",
                "feature_dim": 8,
                "robot_state_dim": 32,
                "action_dim": 7,
                "chunk_m": 4,
                "out_dim": 7,
                "draft_history_len": 6,
                "shards": [{"path": "rank000_shard00000.safetensors", "num_samples": 1, "rank": 0}],
            }
        ),
        encoding="utf-8",
    )

    ds = _ShardCacheDataset(run_dir)
    prefix_embs, prefix_pad_masks, prefix_att_masks, robot, last, tgt = ds[0]

    assert prefix_embs.shape == (16, 8)
    assert prefix_pad_masks.shape == (16,)
    assert prefix_att_masks.shape == (16,)
    assert robot.shape == (32,)
    assert last.shape == (6, 7)
    assert tgt.shape == (4, 7)


def test_shard_cache_dataset_no_longer_falls_back_to_legacy_front_wrist_inputs(tmp_path) -> None:
    run_dir = tmp_path / "legacy_cache"
    run_dir.mkdir()
    save_safetensors(
        {
            "front_feat": torch.zeros((1, 8), dtype=torch.float32),
            "wrist_feat": torch.zeros((1, 8), dtype=torch.float32),
            "robot_state": torch.randn((1, 32), dtype=torch.float32),
            "last_actions": torch.randn((1, 6, 7), dtype=torch.float32),
            "seed_actions": torch.randn((1, 2, 7), dtype=torch.float32),
            "targets": torch.randn((1, 4, 7), dtype=torch.float32),
        },
        str(run_dir / "rank000_shard00000.safetensors"),
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "target_source": "teacher_zero_noise",
                "teacher_noise_mode": "zero",
                "sample_semantics": "sliding_chunk_shift",
                "draft_arch": "vlm_block",
                "draft_input_mode": "prefix_embs",
                "feature_dim": 8,
                "robot_state_dim": 32,
                "action_dim": 7,
                "chunk_m": 4,
                "out_dim": 7,
                "shards": [{"path": "rank000_shard00000.safetensors", "num_samples": 1, "rank": 0}],
            }
        ),
        encoding="utf-8",
    )

    ds = _ShardCacheDataset(run_dir)

    with pytest.raises((KeyError, ValueError), match="prefix"):
        ds[0]


def test_enc_cache_shard_writer_writes_prefix_only_fields(tmp_path) -> None:
    writer = enc_cache._ShardWriter(
        run_dir=tmp_path,
        rank=0,
        shard_size=4,
        cache_dtype=torch.float32,
        history_len=6,
        chunk_m=4,
        out_dim=7,
    )
    writer.add_batch(
        {
            "prefix_embs": torch.randn((2, 16, 8), dtype=torch.float32),
            "prefix_pad_masks": torch.ones((2, 16), dtype=torch.bool),
            "prefix_att_masks": torch.zeros((2, 16), dtype=torch.bool),
            "robot_state": torch.randn((2, 32), dtype=torch.float32),
            "last_actions": torch.randn((2, 6, 7), dtype=torch.float32),
            "seed_actions": torch.randn((2, 2, 7), dtype=torch.float32),
            "targets": torch.randn((2, 4, 7), dtype=torch.float32),
            "episode_index": torch.zeros((2,), dtype=torch.int64),
            "dataset_index": torch.arange(2, dtype=torch.int64),
        }
    )
    writer.finalize()

    shard_path = tmp_path / writer.shards[0]["path"]
    tensors = load_safetensors(str(shard_path))

    assert "prefix_embs" in tensors
    assert "prefix_pad_masks" in tensors
    assert "prefix_att_masks" in tensors
    assert "front_feat" not in tensors
    assert "wrist_feat" not in tensors


def test_enc_cache_load_rank_resume_state_skips_truncated_tail_and_recovers_sliding_window(tmp_path) -> None:
    run_dir = tmp_path / "resume"
    run_dir.mkdir()

    save_safetensors(
        {
            "prefix_embs": torch.randn((2, 4, 8), dtype=torch.float32),
            "prefix_pad_masks": torch.ones((2, 4), dtype=torch.bool),
            "prefix_att_masks": torch.zeros((2, 4), dtype=torch.bool),
            "robot_state": torch.randn((2, 32), dtype=torch.float32),
            "last_actions": torch.tensor(
                [
                    [[0.0] * 7, [1.0] * 7, [2.0] * 7],
                    [[10.0] * 7, [11.0] * 7, [12.0] * 7],
                ],
                dtype=torch.float32,
            ),
            "seed_actions": torch.zeros((2, 2, 7), dtype=torch.float32),
            "targets": torch.tensor(
                [
                    [[20.0] * 7, [21.0] * 7],
                    [[30.0] * 7, [31.0] * 7],
                ],
                dtype=torch.float32,
            ),
            "episode_index": torch.tensor([5, 5], dtype=torch.int64),
            "dataset_index": torch.tensor([0, 1], dtype=torch.int64),
        },
        str(run_dir / "rank000_shard00000.safetensors"),
    )
    (run_dir / "rank000_shard00001.safetensors").write_bytes(b"incomplete")

    resume = enc_cache._load_rank_resume_state(run_dir, rank=0, history_len=3, action_dim=7)

    assert resume.next_shard_id == 1
    assert resume.processed_indices == {0, 1}
    assert resume.complete_shards == [{"path": "rank000_shard00000.safetensors", "num_samples": 2, "rank": 0}]
    assert resume.sliding_state.episode_index == 5
    assert torch.allclose(
        resume.sliding_state.last_actions,
        torch.tensor(
            [[11.0] * 7, [12.0] * 7, [30.0] * 7],
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(
        resume.sliding_state.prev_target_chunk,
        torch.tensor([[30.0] * 7, [31.0] * 7], dtype=torch.float32),
    )


def test_enc_cache_shard_writer_can_resume_from_existing_shards(tmp_path) -> None:
    writer = enc_cache._ShardWriter(
        run_dir=tmp_path,
        rank=0,
        shard_size=2,
        cache_dtype=torch.float32,
        history_len=3,
        chunk_m=2,
        out_dim=7,
        existing_shards=[{"path": "rank000_shard00000.safetensors", "num_samples": 2, "rank": 0}],
        start_shard_id=1,
    )
    writer.add_batch(
        {
            "prefix_embs": torch.randn((1, 4, 8), dtype=torch.float32),
            "prefix_pad_masks": torch.ones((1, 4), dtype=torch.bool),
            "prefix_att_masks": torch.zeros((1, 4), dtype=torch.bool),
            "robot_state": torch.randn((1, 32), dtype=torch.float32),
            "last_actions": torch.randn((1, 3, 7), dtype=torch.float32),
            "seed_actions": torch.randn((1, 2, 7), dtype=torch.float32),
            "targets": torch.randn((1, 2, 7), dtype=torch.float32),
            "episode_index": torch.tensor([0], dtype=torch.int64),
            "dataset_index": torch.tensor([2], dtype=torch.int64),
        }
    )
    writer.finalize()

    assert writer.shards[0]["path"] == "rank000_shard00000.safetensors"
    assert writer.shards[1]["path"] == "rank000_shard00001.safetensors"
    assert (tmp_path / "rank000_shard00001.safetensors").is_file()


def test_enc_cache_skips_corrupt_video_and_reports_failed_path(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_data_error = type("InvalidDataError", (Exception,), {})
    invalid_data_error.__module__ = "av.error"

    writes: list[str] = []

    class _Meta:
        video_keys = ["observation.images.wrist_image", "observation.images.image"]
        tasks = ["push the plate to the front of the stove"]

        @staticmethod
        def get_video_file_path(ep_idx: int, vid_key: str) -> str:
            return f"videos/chunk-000/{vid_key}/episode_{ep_idx:06d}.mp4"

    class _HFDataset:
        @staticmethod
        def __getitem__(idx: int) -> dict[str, object]:
            del idx
            return {"episode_index": 82, "task_index": 0, "timestamp": 5.25}

    class _Dataset:
        root = Path("/dataset")
        meta = _Meta()
        delta_indices = {"action": [0]}
        tolerance_s = 0.05
        video_backend = "pyav"
        hf_dataset = _HFDataset()

        def __getitem__(self, idx: int) -> dict[str, object]:
            del idx
            raise invalid_data_error("Invalid data found when processing input: 'avcodec_send_packet()'")

        @staticmethod
        def _get_query_indices(idx: int, ep_idx: int):
            return {"action": [idx]}, {"episode_index": ep_idx}

        def _get_query_timestamps(self, current_ts: float, query_indices: dict[str, list[int]]) -> dict[str, list[float]]:
            del query_indices
            return {vid_key: [float(current_ts)] for vid_key in self.meta.video_keys}

    def _fake_decode(video_path, timestamps, tolerance_s, backend):
        del timestamps, tolerance_s, backend
        if "wrist_image" in str(video_path):
            raise invalid_data_error("Invalid data found when processing input: 'avcodec_send_packet()'")
        return torch.zeros((1, 3, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(enc_cache, "decode_video_frames", _fake_decode)
    monkeypatch.setattr(enc_cache.tqdm.tqdm, "write", lambda msg: writes.append(str(msg)))

    item = enc_cache._load_dataset_item_or_skip(_Dataset(), 5327, rank=0)

    assert item is None
    assert writes
    message = "\n".join(writes)
    assert "skip_corrupt_video" in message
    assert "dataset_index=5327" in message
    assert "episode=82" in message
    assert "task=push the plate to the front of the stove" in message
    assert "observation.images.wrist_image" in message
    assert "episode_000082.mp4" in message
    assert "failed_video" in message


@pytest.mark.parametrize("target_source", ["teacher_zero_noise", "gt"])
def test_enc_cache_targets_from_observation_accept_current_encoder_tuple(target_source: str) -> None:
    prefix_embs = torch.randn((1, 5, 8), dtype=torch.float32)
    prefix_pad_masks = torch.ones((1, 5), dtype=torch.bool)
    prefix_att_masks = torch.zeros((1, 5), dtype=torch.bool)
    robot_state = torch.randn((1, 32), dtype=torch.float32)
    teacher_actions = torch.full((1, 6, 7), 2.0, dtype=torch.float32)
    gt_targets = torch.full((1, 6, 7), 4.0, dtype=torch.float32)

    class _Config:
        action_horizon = 6
        action_dim = 7

    class _SpecArgsStub:
        full_num_steps = 3

    class _SpecModelStub:
        config = _Config()
        spec_args = _SpecArgsStub()

        def _preprocess_observation(self, observation, *, train=False):
            del observation, train
            return (
                (torch.ones((1, 2, 8), dtype=torch.float32),),
                (torch.ones((1,), dtype=torch.bool),),
                torch.ones((1, 1, 8), dtype=torch.float32),
                torch.ones((1, 1), dtype=torch.bool),
                robot_state,
            )

        def _encoder_stage(self, images_t, img_masks_t, lang_tokens, lang_masks):
            del images_t, img_masks_t, lang_tokens, lang_masks
            return prefix_embs, prefix_pad_masks, prefix_att_masks

        def _vlm_prefill_stage(self, prefix_embs_in, prefix_pad_masks_in, prefix_att_masks_in):
            assert torch.equal(prefix_embs_in, prefix_embs)
            assert torch.equal(prefix_pad_masks_in, prefix_pad_masks)
            assert torch.equal(prefix_att_masks_in, prefix_att_masks)
            return object()

        def _full_action_stage(self, robot_state_in, prefix_pad_masks_in, past_key_values, zero_noise, full_num_steps):
            assert torch.equal(robot_state_in, robot_state)
            assert torch.equal(prefix_pad_masks_in, prefix_pad_masks)
            assert past_key_values is not None
            assert tuple(zero_noise.shape) == (1, 6, 7)
            assert full_num_steps == 3
            return teacher_actions

    prefix_embs_out, prefix_pad_masks_out, prefix_att_masks_out, robot_state_out, targets = enc_cache._targets_from_observation(
        _SpecModelStub(),
        observation=object(),
        device=torch.device("cpu"),
        target_source=target_source,
        chunk_m=4,
        out_dim=7,
        gt_targets=gt_targets if target_source == "gt" else None,
    )

    assert torch.equal(prefix_embs_out, prefix_embs)
    assert torch.equal(prefix_pad_masks_out, prefix_pad_masks)
    assert torch.equal(prefix_att_masks_out, prefix_att_masks)
    assert torch.equal(robot_state_out, robot_state)
    assert tuple(targets.shape) == (1, 4, 7)
    expected_value = 2.0 if target_source == "teacher_zero_noise" else 4.0
    assert torch.allclose(targets, torch.full((1, 4, 7), expected_value, dtype=torch.float32))


def test_training_rejects_non_prefix_cache_for_vlm_block_head() -> None:
    head = DraftChunkHead(
        img_dim=8,
        chunk_m=4,
        hidden_dim=16,
        out_dim=7,
    )
    ds = type("DatasetStub", (), {"draft_input_mode": "spatial_tokens"})()

    with pytest.raises(ValueError, match="prefix-embedding cache"):
        _require_cache_compatible_with_head(ds)


def test_runtime_make_draft_head_ignores_legacy_binary_gripper_meta() -> None:
    class _LanguageModel:
        config = type(
            "Config",
            (),
            {
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
            },
        )()
        layers: list[object] = []

    class _PaliGemma:
        language_model = _LanguageModel()

    class _WithExpert:
        paligemma = _PaliGemma()

    class _RuntimeStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
            self._chunk_m = 2
            self.paligemma_with_expert = _WithExpert()

    head = SpecPI0Pytorch._make_draft_head(
        _RuntimeStub(),
        img_dim=8,
        device=torch.device("cpu"),
        meta={"draft_gripper_mode": "binary"},
    )
    assert isinstance(head, DraftChunkHead)

    with torch.no_grad():
        for param in head.parameters():
            param.zero_()
        head._action_head.bias[6] = 0.25

    actions = head(
        prefix_embs=torch.zeros((1, 3, 8), dtype=torch.float32),
        prefix_pad_masks=torch.ones((1, 3), dtype=torch.bool),
        prefix_att_masks=torch.zeros((1, 3), dtype=torch.bool),
        robot_state=torch.zeros((1, 32), dtype=torch.float32),
        last_actions=torch.zeros((1, 6, 7), dtype=torch.float32),
    )

    assert actions.shape == (1, 2, 7)
    assert torch.allclose(actions[:, :, 6], torch.full((1, 2), 0.25, dtype=torch.float32))


def test_runtime_load_draft_head_rejects_pooled_checkpoint(tmp_path) -> None:
    ckpt_path = tmp_path / "old_pooled.pt"
    torch.save(
        {
            "meta": {"draft_arch": "pooled"},
            "draft_head": {"_gru.weight_ih_l0": torch.zeros((1, 1), dtype=torch.float32)},
        },
        ckpt_path,
    )

    class _VisionConfig:
        projection_dim = 8

    class _PaliGemmaConfig:
        vision_config = _VisionConfig()

    class _PaliGemma:
        config = _PaliGemmaConfig()

    class _WithExpert:
        paligemma = _PaliGemma()

    class _RuntimeStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
            self._draft_head = torch.nn.Identity()
            self.paligemma_with_expert = _WithExpert()

        def _make_draft_head(self, **_kwargs) -> DraftChunkHead:
            return DraftChunkHead(
                img_dim=8,
                chunk_m=4,
                hidden_dim=16,
                out_dim=7,
            )

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        SpecPI0Pytorch.load_draft_head(_RuntimeStub(), str(ckpt_path))
