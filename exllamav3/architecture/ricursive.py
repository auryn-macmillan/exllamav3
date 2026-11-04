from __future__ import annotations

from typing_extensions import override

import torch

from ..cache.recurrent_util import prepare_for_recurrence
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    Attention,
    Embedding,
    GatedMLP,
    Linear,
    RMSNorm,
    SlidingAttention,
    SWAState,
    TransformerBlock,
)
from ..modules.attn import prepare_for_attn
from ..util.rope import RopeStyle


class RicursiveConfig(Config):
    """
    Ricursive Intelligence "Ripple" architecture.

    Thin-and-deep Gemma-3N-style blocks: PeriNorm (pre/post attention and pre/post FFN
    norms), elementwise sigmoid-gated attention, per-head Q/K RMSNorm, SwiGLU with an
    activation clamp, a 4:1 sliding-window/global pattern (global layers have no RoPE),
    an embedding scale carried in config.json, and a learned per-layer value residual.
    """

    arch_string = "RicursiveForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": RicursiveModel},
            **kwargs,
        )

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        self.sliding_window = self.read_cfg(int, "sliding_window", 1024)
        layer_types = self.read_cfg(list, "layer_types", None)
        if layer_types:
            assert len(layer_types) == self.num_hidden_layers, \
                "Length of layer_types does not match num_hidden_layers"
            self.swa_pattern = []
            for t in layer_types:
                if t == "sliding_attention":
                    # HF keeps sliding_window keys including the query; the kernel's left
                    # window is "keys before the query", so pass window - 1
                    self.swa_pattern.append(self.sliding_window - 1)
                elif t == "full_attention":
                    self.swa_pattern.append(-1)
                else:
                    raise ValueError(f"Unknown Ricursive layer type: {t}")
        else:
            self.swa_pattern = [
                -1 if (idx + 1) % 5 == 0 else self.sliding_window - 1
                for idx in range(self.num_hidden_layers)
            ]

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        # Reference clamps gate to +limit and up to +/-limit. 0 disables the clamp
        self.ffn_clamp_threshold = self.read_cfg(float, "ffn_clamp_threshold", 0.0) or 0.0

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", 1e-6)

        # RoPE is only used on the sliding-attention layers. The global layers are NoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)

        # Embedding scale: non-folded checkpoints carry sqrt(hidden_size) here and scale at
        # runtime; folded checkpoints leave it null and bake the scale into the table
        self.embed_scale = self.read_cfg(float, "embed_scale", None)

        # Attention/value-residual switches
        self.use_gated_attention = self.read_cfg(bool, "use_gated_attention", True)
        self.use_valres = self.read_cfg(bool, "use_valres", False)

    def default_max_position_embeddings(self):
        return 2097152


class RicursiveModel(Model):
    config_class = RicursiveConfig

    def __init__(
        self,
        config: RicursiveConfig,
        key_prefix: str = "model",
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
                multiplier = config.embed_scale if config.embed_scale is not None else 1.0,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            is_swa = config.swa_pattern[idx] >= 0
            rope_settings = config.rope_settings if is_swa else None
            attn_cls = SlidingAttention if is_swa else Attention

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn_cls(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.self_attn",
                        layer_idx = idx,
                        hidden_size = config.hidden_size,
                        head_dim = config.head_dim,
                        num_q_heads = config.num_q_heads,
                        num_kv_heads = config.num_kv_heads,
                        rope_settings = rope_settings,
                        sm_scale = None,
                        sliding_window = config.swa_pattern[idx] if is_swa else -1,
                        key_q = "q_proj",
                        key_k = "k_proj",
                        key_v = "v_proj",
                        key_o = "o_proj",
                        key_g = "g_proj" if config.use_gated_attention else None,
                        full_gate = True,
                        valres_key = "value_res_lambda" if config.use_valres else None,
                        qmap = "block.attn",
                        q_norm = RMSNorm(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.self_attn.q_norm",
                            rms_norm_eps = config.rms_norm_eps,
                        ),
                        k_norm = RMSNorm(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.self_attn.k_norm",
                            rms_norm_eps = config.rms_norm_eps,
                        ),
                        out_dtype = torch.float,
                    ),
                    attn_post_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.pre_feedforward_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        act_limit = config.ffn_clamp_threshold,
                    ),
                    mlp_post_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_feedforward_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{key_prefix}.embed_tokens"

        self.modules += [
            RMSNorm(
                config = config,
                key = f"{key_prefix}.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True},
            ),
        ]

        self.logit_layer_idx = len(self.modules) - 1

        # SWA layers are recurrent: the generator allocates one SWAState per sequence.
        # Checkpoints are expensive, so use the same longer interval as the other SWA models
        self.recurrent_state_cls = None
        if any(p >= 0 for p in config.swa_pattern):
            self.caps.update({
                "recurrent_states": True,
                "default_recurrent_checkpoint_interval": 6144,
            })
            self.recurrent_state_cls = SWAState

        # Value residuals carry the layer-0 embedding into every layer; TP would need to
        # replicate that seed on every rank, so leave TP off until it is plumbed through
        self.caps.update({"supports_tp": False})

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        prepare_for_recurrence(input_ids, params, self)
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        p += f"<|im_start|>user\n{prompt}<|im_end|>\n"
        p += "<|im_start|>assistant\n"
        return p
