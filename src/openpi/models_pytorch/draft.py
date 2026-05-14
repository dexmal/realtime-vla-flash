import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding


class DraftChunkHead(nn.Module):
    """draft head: one-layer Gemma query decoder over prefix embeddings."""

    def __init__(
        self,
        *,
        img_dim: int,
        chunk_m: int,
        hidden_dim: int = 256,
        out_dim: int = 7,
        num_heads: int | None = None,
        num_kv_heads: int = 1,
        head_dim: int | None = None,
        dtype: torch.dtype = torch.float32,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()
        self.chunk_m = int(chunk_m)
        self.out_dim = int(out_dim)
        self.pose_rot_dim = int(min(6, self.out_dim))
        self.hidden_size = int(img_dim)
        self.num_heads = int(num_heads or self._resolve_num_heads(self.hidden_size))
        self.num_kv_heads = int(max(1, int(num_kv_heads)))
        self.head_dim = int(head_dim or max(1, self.hidden_size // self.num_heads))
        self.attn_implementation = str(attn_implementation)

        gemma_config = CONFIG_MAPPING["gemma"](
            head_dim=int(self.head_dim),
            hidden_size=int(self.hidden_size),
            intermediate_size=int(hidden_dim),
            num_attention_heads=int(self.num_heads),
            num_hidden_layers=1,
            num_key_value_heads=int(self.num_kv_heads),
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype=str(dtype).replace("torch.", ""),
        )
        gemma_config._attn_implementation = self.attn_implementation  # noqa: SLF001

        self._state_token = nn.Linear(32, int(self.hidden_size))
        self._action_queries = nn.Embedding(int(self.chunk_m), int(self.hidden_size))
        self._gemma_block = GemmaDecoderLayer(gemma_config, layer_idx=0)
        self._rotary_emb = GemmaRotaryEmbedding(gemma_config)
        self._action_head = nn.Linear(int(self.hidden_size), int(self.out_dim))

    @staticmethod
    def _resolve_num_heads(dim: int) -> int:
        for heads in (8, 4, 2, 1):
            if dim % heads == 0:
                return heads
        return 1

    @staticmethod
    def _make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
        cumsum = torch.cumsum(att_masks.to(dtype=torch.int64), dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    def _build_attention_mask(self, *, prefix_pad_masks: torch.Tensor, prefix_att_masks: torch.Tensor | None = None) -> torch.Tensor:
        if prefix_pad_masks.ndim != 2:
            raise ValueError(f"expected prefix_pad_masks to be (B,S), got shape={tuple(prefix_pad_masks.shape)}")
        b, s = int(prefix_pad_masks.shape[0]), int(prefix_pad_masks.shape[1])
        device = prefix_pad_masks.device
        if prefix_att_masks is None:
            prefix_att_masks = torch.zeros((b, s), device=device, dtype=torch.bool)
        if prefix_att_masks.ndim != 2 or tuple(prefix_att_masks.shape) != tuple(prefix_pad_masks.shape):
            raise ValueError(
                f"expected prefix_att_masks to match prefix_pad_masks shape={tuple(prefix_pad_masks.shape)}, got {tuple(prefix_att_masks.shape)}"
            )

        state_pad = torch.ones((b, 1), device=device, dtype=torch.bool)
        state_att = torch.zeros((b, 1), device=device, dtype=torch.bool)
        prefix_plus_state_pad = torch.cat([prefix_pad_masks.to(dtype=torch.bool), state_pad], dim=1)
        prefix_plus_state_att = torch.cat([prefix_att_masks.to(dtype=torch.bool), state_att], dim=1)
        prefix_mask = self._make_att_2d_masks(prefix_plus_state_pad, prefix_plus_state_att)

        query_count = int(self.chunk_m)
        total = int(prefix_mask.shape[1] + query_count)
        mask = torch.zeros((b, total, total), device=device, dtype=torch.bool)
        prefix_len = int(prefix_mask.shape[1])
        mask[:, :prefix_len, :prefix_len] = prefix_mask
        mask[:, prefix_len:, :prefix_len] = prefix_plus_state_pad[:, None, :]
        mask[:, prefix_len:, prefix_len:] = True
        return mask

    def _build_position_ids(self, *, prefix_pad_masks: torch.Tensor) -> torch.Tensor:
        b = int(prefix_pad_masks.shape[0])
        state_pad = torch.ones((b, 1), device=prefix_pad_masks.device, dtype=torch.bool)
        query_pad = torch.ones((b, int(self.chunk_m)), device=prefix_pad_masks.device, dtype=torch.bool)
        pad_mask = torch.cat([prefix_pad_masks.to(dtype=torch.bool), state_pad, query_pad], dim=1)
        return (torch.cumsum(pad_mask.to(dtype=torch.int64), dim=1) - 1).clamp_min(0)

    def init_from_vlm_layer(self, layer: nn.Module) -> None:
        self._gemma_block.load_state_dict(layer.state_dict(), strict=True)

    def forward(
        self,
        *,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        robot_state: torch.Tensor,
        last_actions: torch.Tensor,
    ) -> torch.Tensor:
        del last_actions
        if prefix_embs.ndim != 3:
            raise ValueError(f"expected prefix_embs to be (B,S,H), got shape={tuple(prefix_embs.shape)}")
        if prefix_pad_masks.ndim != 2:
            raise ValueError(f"expected prefix_pad_masks to be (B,S), got shape={tuple(prefix_pad_masks.shape)}")
        if prefix_att_masks.ndim != 2:
            raise ValueError(f"expected prefix_att_masks to be (B,S), got shape={tuple(prefix_att_masks.shape)}")
        if robot_state.ndim != 2 or int(robot_state.shape[1]) != 32:
            raise ValueError(f"expected robot_state to be (B,32), got shape={tuple(robot_state.shape)}")

        if int(prefix_embs.shape[0]) != int(robot_state.shape[0]):
            raise ValueError("prefix_embs and robot_state must have matching batch dimensions")
        if int(prefix_embs.shape[1]) != int(prefix_pad_masks.shape[1]) or int(prefix_embs.shape[1]) != int(prefix_att_masks.shape[1]):
            raise ValueError("prefix_embs, prefix_pad_masks, and prefix_att_masks must have matching sequence lengths")
        if int(prefix_embs.shape[2]) != int(self.hidden_size):
            raise ValueError(f"expected prefix_embs hidden size={self.hidden_size}, got {int(prefix_embs.shape[2])}")

        b = int(robot_state.shape[0])
        block_dtype = self._gemma_block.self_attn.q_proj.weight.dtype
        prefix_embs = prefix_embs.to(dtype=block_dtype)
        state_token = self._state_token(robot_state.to(dtype=self._state_token.weight.dtype))[:, None, :].to(dtype=block_dtype)
        query_ids = torch.arange(int(self.chunk_m), device=prefix_embs.device, dtype=torch.long)[None, :].expand(b, -1)
        query_tokens = self._action_queries(query_ids).to(dtype=block_dtype)
        hidden_states = torch.cat([prefix_embs, state_token, query_tokens], dim=1)

        mask_2d = self._build_attention_mask(prefix_pad_masks=prefix_pad_masks, prefix_att_masks=prefix_att_masks)
        attention_mask = torch.where(
            mask_2d[:, None, :, :],
            torch.zeros((), device=hidden_states.device, dtype=block_dtype),
            torch.full((), torch.finfo(block_dtype).min, device=hidden_states.device, dtype=block_dtype),
        )
        position_ids = self._build_position_ids(prefix_pad_masks=prefix_pad_masks)
        position_embeddings = self._rotary_emb(hidden_states, position_ids)
        hidden_states = self._gemma_block(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=position_embeddings,
            adarms_cond=None,
        )[0]

        query_hidden = hidden_states[:, -int(self.chunk_m) :, :].to(dtype=self._action_head.weight.dtype)
        actions = self._action_head(query_hidden).to(dtype=torch.float32)
        return actions
