from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from openpi.models_pytorch.draft import DraftChunkHead
from scripts.spec.triton import triton_pi0_runtime as triton_runtime


def _load_pi0_spec_infer_module():
    module_path = Path(__file__).resolve().parents[1] / "triton" / "pi0_spec_infer.py"
    spec = importlib.util.spec_from_file_location("pi0_spec_infer_draft_kernel_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_triton_draft_kernel_matches_reference_head(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    img_dim = 8
    chunk_m = 4
    out_dim = 7
    hidden_dim = 16
    num_heads = 2
    num_kv_heads = 1
    head_dim = 4

    reference = DraftChunkHead(
        img_dim=img_dim,
        chunk_m=chunk_m,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device=device, dtype=dtype)
    reference.eval()

    state_dict = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
    checkpoint_path = tmp_path / "draft_head.pt"
    torch.save(
        {
            "meta": {
                "img_dim": img_dim,
                "chunk_m": chunk_m,
                "out_dim": out_dim,
                "draft_num_heads": num_heads,
                "draft_num_kv_heads": num_kv_heads,
                "draft_head_dim": head_dim,
            },
            "draft_head": state_dict,
        },
        checkpoint_path,
    )

    converted_path = triton_runtime.convert_spec_draft_checkpoint(
        draft_checkpoint_path=checkpoint_path,
        output_path=tmp_path / "draft_triton.pkl",
    )
    with converted_path.open("rb") as handle:
        draft_artifact = triton_runtime.pickle.load(handle)

    module = _load_pi0_spec_infer_module()
    runtime = module.Pi0SpecInference(
        checkpoint={"language_embeds": torch.zeros((2, img_dim), dtype=dtype)},
        draft_checkpoint=draft_artifact,
        num_views=2,
        chunk_size=50,
    )

    batch_size = 1
    prefix_len = 5
    prefix_embs = torch.randn((batch_size, prefix_len, img_dim), device=device, dtype=dtype)
    prefix_pad_masks = torch.ones((batch_size, prefix_len), device=device, dtype=torch.bool)
    prefix_att_masks = torch.zeros((batch_size, prefix_len), device=device, dtype=torch.bool)
    robot_state = torch.randn((batch_size, 32), device=device, dtype=dtype)
    last_actions = torch.randn((batch_size, 6, 32), device=device, dtype=dtype)

    with torch.inference_mode():
        reference_actions = reference(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            robot_state=robot_state,
            last_actions=last_actions,
        ).to(dtype=torch.float32)

        prefix = module.PrefixContext(
            prefix_embs=prefix_embs.to(dtype=torch.float32),
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
        )
        triton_actions = runtime._run_draft_block(
            prefix=prefix,
            observation_state_normalized=robot_state.to(dtype=torch.float32),
        )[:, :chunk_m, :out_dim]

    max_diff = float((reference_actions - triton_actions).abs().max().item())
    assert max_diff < 1e-2, f"draft kernel drift too large: max_diff={max_diff}"
