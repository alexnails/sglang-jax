import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding, _yarn_get_mscale
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK, create_moe_weights_mapping
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def _get_rope_scaling(config: PretrainedConfig) -> dict[str, Any] | None:
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return None
    if isinstance(rope_scaling, dict):
        rope_scaling = dict(rope_scaling)
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
        if rope_type in (None, "default"):
            return None
    return rope_scaling


def _get_deepseek_softmax_scale(q_head_dim: int, rope_scaling: dict[str, Any] | None) -> float:
    scale = q_head_dim**-0.5
    if rope_scaling is None:
        return scale
    if rope_scaling.get("type", rope_scaling.get("rope_type")) == "yarn":
        factor = float(rope_scaling.get("factor", 1.0))
        if rope_scaling.get("mscale_all_dim", 0):
            mscale = float(_yarn_get_mscale(factor))
            scale *= mscale * mscale
    return scale


def _scale_source_keys(weight_key: str) -> list[str]:
    return [
        weight_key.replace(".weight", ".weight_scale_inv"),
        weight_key.replace(".weight", ".weight_scale"),
    ]


class DeepseekV3MLP(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        self.layer_id = layer_id
        self.gate_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.up_proj = LinearBase(
            input_size=hidden_size,
            output_size=intermediate_size,
            kernel_axes=(None, "tensor"),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.down_proj = LinearBase(
            input_size=intermediate_size,
            output_size=hidden_size,
            kernel_axes=("tensor", None),
            use_bias=False,
            params_dtype=dtype,
            mesh=mesh,
        )
        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        out, _ = self.down_proj(self.act_fn(gate) * up)
        return out


class DeepseekV3Attention(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.q_lora_rank is None:
            self.q_proj = LinearBase(
                input_size=self.hidden_size,
                output_size=self.num_heads * self.q_head_dim,
                use_bias=False,
                kernel_axes=(None, "tensor"),
                params_dtype=dtype,
                mesh=mesh,
            )
        else:
            self.q_a_proj = LinearBase(
                input_size=self.hidden_size,
                output_size=self.q_lora_rank,
                use_bias=getattr(config, "attention_bias", False),
                kernel_axes=(None, "tensor"),
                params_dtype=dtype,
                mesh=mesh,
            )
            self.q_a_layernorm = RMSNorm(
                self.q_lora_rank,
                epsilon=config.rms_norm_eps,
                param_dtype=dtype,
            )
            self.q_b_proj = LinearBase(
                input_size=self.q_lora_rank,
                output_size=self.num_heads * self.q_head_dim,
                use_bias=False,
                kernel_axes=(None, "tensor"),
                params_dtype=dtype,
                mesh=mesh,
            )

        self.kv_a_proj_with_mqa = LinearBase(
            input_size=self.hidden_size,
            output_size=self.kv_lora_rank + self.qk_rope_head_dim,
            use_bias=getattr(config, "attention_bias", False),
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )
        self.kv_b_proj = LinearBase(
            input_size=self.kv_lora_rank,
            output_size=self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.o_proj = LinearBase(
            input_size=self.num_heads * self.v_head_dim,
            output_size=self.hidden_size,
            use_bias=getattr(config, "attention_bias", False),
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )

        max_position_embeddings = getattr(config, "max_position_embeddings", 4096)
        rope_theta = getattr(config, "rope_theta", 10000)
        self.rotary_emb = RotaryEmbedding(
            head_size=self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )
        rope_scaling = _get_rope_scaling(config)
        self.scaling = _get_deepseek_softmax_scale(self.q_head_dim, rope_scaling)
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.q_head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, Any]:
        if self.q_lora_rank is None:
            q, _ = self.q_proj(hidden_states)
        else:
            q_lora, _ = self.q_a_proj(hidden_states)
            q = self.q_a_layernorm(q_lora)
            q, _ = self.q_b_proj(q)
        q = q.reshape(-1, self.num_heads, self.q_head_dim)
        q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=-1)

        compressed_kv, _ = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = jnp.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.reshape(-1, 1, self.qk_rope_head_dim)

        kv, _ = self.kv_b_proj(compressed_kv)
        kv = kv.reshape(-1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = jnp.split(kv, [self.qk_nope_head_dim], axis=-1)

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_full = jnp.concatenate([q_nope, q_pe], axis=-1)
        attn_output, kv_state = self.attn(
            q_full,
            k_nope,
            v,
            forward_batch,
            token_to_kv_pool,
            k_pe=k_pe,
        )
        output, _ = self.o_proj(attn_output)
        return output, kv_state


class DeepseekV3MoE(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        self.num_routed_experts = config.n_routed_experts
        self.num_shared_experts = getattr(config, "n_shared_experts", 0) or 0
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_backend = getattr(config, "moe_backend", "epmoe")
        self.use_fused = self.moe_backend == "fused"

        self.moe_gate = GateLogit(
            input_size=config.hidden_size,
            num_experts=self.num_routed_experts,
            weight_dtype=jnp.float32,
            enable_expert_bias=getattr(config, "topk_method", "") == "noaux_tc",
            score_func=getattr(config, "scoring_func", "sigmoid"),
        )
        self.topk = TopK(
            topk=self.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            num_expert_group=getattr(config, "n_group", 0),
            topk_group=getattr(config, "topk_group", 0),
            routed_scaling_factor=getattr(config, "routed_scaling_factor", None),
            layer_id=layer_id,
        )

        if self.use_fused:
            self.mlp = FusedEPMoE(
                hidden_size=config.hidden_size,
                num_experts=self.num_routed_experts,
                num_experts_per_tok=self.num_experts_per_tok,
                intermediate_dim=config.moe_intermediate_size,
                mesh=mesh,
                ep_size=config.ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                renormalize_topk_logits=config.norm_topk_prob,
                routed_scaling_factor=getattr(config, "routed_scaling_factor", None),
                use_grouped_topk=getattr(config, "n_group", 0) > 0,
                num_groups=getattr(config, "n_group", 0),
                top_k_groups=getattr(config, "topk_group", 0),
                num_shared_experts=self.num_shared_experts,
                moe_shared_expert_intermediate_size=config.moe_intermediate_size,
                quantization_config=getattr(config, "quantization_config", None),
            )
            self.shared_experts = None
        else:
            self.mlp = EPMoE(
                hidden_size=config.hidden_size,
                num_experts=self.num_routed_experts,
                num_experts_per_tok=self.num_experts_per_tok,
                intermediate_dim=config.moe_intermediate_size,
                mesh=mesh,
                ep_size=config.ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                quantization_config=getattr(config, "quantization_config", None),
            )
            if self.num_shared_experts > 0:
                self.shared_experts = DeepseekV3MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size * self.num_shared_experts,
                    layer_id=layer_id,
                    dtype=dtype,
                    mesh=mesh,
                )
            else:
                self.shared_experts = None

    def __call__(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        dispatch_info: ExpertLocationMetadata | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        correction_bias = self.moe_gate.bias.value if self.moe_gate.bias is not None else None
        router_logits = self.moe_gate(hidden_states)
        topk_weights, topk_ids = self.topk(
            router_logits,
            correction_bias=correction_bias,
            dispatch_info=dispatch_info,
        )

        if self.use_fused:
            token_valid_mask = forward_batch.get_token_valid_mask(hidden_states.shape[0])
            topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)
        mlp_output = self.mlp(hidden_states, topk_weights, topk_ids)

        if self.shared_experts is not None:
            mlp_output = mlp_output + self.shared_experts(hidden_states)

        return mlp_output, topk_ids


class DeepseekV3DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.self_attn = DeepseekV3Attention(
            config=config,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )

        use_moe = (
            getattr(config, "n_routed_experts", None) is not None
            and layer_id >= getattr(config, "first_k_dense_replace", 0)
            and ((layer_id - getattr(config, "first_k_dense_replace", 0)) % getattr(config, "moe_layer_freq", 1) == 0)
        )
        if use_moe:
            self.mlp = DeepseekV3MoE(
                config=config,
                layer_id=layer_id,
                dtype=dtype,
                mesh=mesh,
            )
            self.is_moe_layer = True
        else:
            self.mlp = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                mesh=mesh,
            )
            self.is_moe_layer = False

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_state = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            hidden_states, topk_ids = self.mlp(
                hidden_states,
                forward_batch,
                dispatch_info=dispatch_info,
            )
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return hidden_states, residual, kv_state, topk_ids


class DeepseekV3Model(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                DeepseekV3DecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list[Any], list[jax.Array | None]]:
        input_embeds = (
            forward_batch.input_embedding
            if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            else None
        )
        hidden_states = (
            self.embed_tokens(forward_batch.input_ids) if input_embeds is None else input_embeds
        )
        residual = None
        layers_kv_state = []
        layers_topk_ids = []
        for layer in self.layers:
            hidden_states, residual, kv_state, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            layers_kv_state.append(kv_state)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_state, layers_topk_ids


class DeepseekV3ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info("DeepseekV3ForCausalLM config dtype: %s", self.dtype)
        self.model = DeepseekV3Model(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
                mesh=mesh,
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        loader.load_weights_from_safetensors(self._create_weight_mappings())
        logger.info("DeepseekV3 weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        quant_config = getattr(self.config, "quantization_config", None)
        is_static_quant = quant_config is not None and getattr(quant_config, "is_static_checkpoint", False)

        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        first_dense = getattr(self.config, "first_k_dense_replace", 0)
        moe_layer_freq = getattr(self.config, "moe_layer_freq", 1)
        for layer_idx in range(self.config.num_hidden_layers):
            is_moe = (
                getattr(self.config, "n_routed_experts", None) is not None
                and layer_idx >= first_dense
                and ((layer_idx - first_dense) % moe_layer_freq == 0)
            )
            mappings.update(self._create_layer_mappings(layer_idx, is_moe, is_static_quant))

        return mappings

    def _create_layer_mappings(
        self,
        layer_idx: int,
        is_moe_layer: bool,
        is_static_quant: bool = False,
    ) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"
        mappings = {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.self_attn.kv_a_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.self_attn.kv_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

        def add_linear_mapping(
            hf_weight_key: str,
            target_base: str,
            sharding_std: tuple[str | None, ...],
        ):
            if is_static_quant:
                sharding_quant = (
                    (sharding_std[1], sharding_std[0]) if len(sharding_std) == 2 else sharding_std
                )
                mappings[hf_weight_key] = WeightMapping(
                    target_path=f"{target_base}.weight_q",
                    sharding=sharding_quant,
                    transpose=False,
                )
                scale_sharding = (sharding_quant[0],)
                if target_base.endswith(("o_proj", "down_proj")):
                    scale_sharding = (None,)
                for scale_key in _scale_source_keys(hf_weight_key):
                    mappings[scale_key] = WeightMapping(
                        target_path=f"{target_base}.weight_scale",
                        sharding=scale_sharding,
                        transpose=False,
                    )
            else:
                mappings[hf_weight_key] = WeightMapping(
                    target_path=f"{target_base}.weight",
                    sharding=sharding_std,
                    transpose=True,
                )

        if getattr(self.config, "q_lora_rank", None) is None:
            add_linear_mapping(
                f"{prefix}.self_attn.q_proj.weight",
                f"{target_prefix}.self_attn.q_proj",
                (None, "tensor"),
            )
        else:
            add_linear_mapping(
                f"{prefix}.self_attn.q_a_proj.weight",
                f"{target_prefix}.self_attn.q_a_proj",
                (None, "tensor"),
            )
            mappings[f"{prefix}.self_attn.q_a_layernorm.weight"] = WeightMapping(
                target_path=f"{target_prefix}.self_attn.q_a_layernorm.scale",
                sharding=(None,),
                transpose=False,
            )
            add_linear_mapping(
                f"{prefix}.self_attn.q_b_proj.weight",
                f"{target_prefix}.self_attn.q_b_proj",
                (None, "tensor"),
            )

        add_linear_mapping(
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
            f"{target_prefix}.self_attn.kv_a_proj_with_mqa",
            (None, "tensor"),
        )
        add_linear_mapping(
            f"{prefix}.self_attn.kv_b_proj.weight",
            f"{target_prefix}.self_attn.kv_b_proj",
            (None, "tensor"),
        )
        add_linear_mapping(
            f"{prefix}.self_attn.o_proj.weight",
            f"{target_prefix}.self_attn.o_proj",
            ("tensor", None),
        )

        if getattr(self.config, "attention_bias", False):
            bias_mappings = {
                f"{prefix}.self_attn.kv_a_proj_with_mqa.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.kv_a_proj_with_mqa.bias",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.o_proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
            }
            if getattr(self.config, "q_lora_rank", None) is None:
                bias_mappings[f"{prefix}.self_attn.q_proj.bias"] = WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=(None,),
                    transpose=False,
                )
            else:
                bias_mappings[f"{prefix}.self_attn.q_a_proj.bias"] = WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_a_proj.bias",
                    sharding=(None,),
                    transpose=False,
                )
            mappings.update(bias_mappings)

        if not is_moe_layer:
            add_linear_mapping(
                f"{prefix}.mlp.gate_proj.weight",
                f"{target_prefix}.mlp.gate_proj",
                (None, "tensor"),
            )
            add_linear_mapping(
                f"{prefix}.mlp.up_proj.weight",
                f"{target_prefix}.mlp.up_proj",
                (None, "tensor"),
            )
            add_linear_mapping(
                f"{prefix}.mlp.down_proj.weight",
                f"{target_prefix}.mlp.down_proj",
                ("tensor", None),
            )
            return mappings

        mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
            target_path=f"{target_prefix}.mlp.moe_gate.kernel",
            sharding=(None, None),
            transpose=True,
        )
        if getattr(self.config, "topk_method", "") == "noaux_tc":
            mappings[f"{prefix}.mlp.gate.e_score_correction_bias"] = WeightMapping(
                target_path=f"{target_prefix}.mlp.moe_gate.bias",
                sharding=(None,),
                transpose=False,
            )

        moe_backend = getattr(self.config, "moe_backend", "epmoe")
        num_experts = getattr(self.config, "n_routed_experts", 0)
        from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata

        metadata = get_global_expert_location_metadata()
        phy_to_log = None
        if metadata is not None:
            physical_to_logical_map = np.array(jax.device_get(metadata.physical_to_logical_map))
            phy_to_log = physical_to_logical_map[layer_idx]

        moe_mappings = create_moe_weights_mapping(
            prefix=prefix,
            target_prefix=target_prefix,
            num_experts=num_experts,
            moe_backend=moe_backend,
            moe_path="mlp",
            source_expert_pattern="experts.{i}",
            physical_to_logical_map=phy_to_log,
        )
        if is_static_quant:
            hidden_size = self.config.hidden_size
            inter_size = getattr(self.config, "moe_intermediate_size", 2048)
            num_physical_experts = num_experts if phy_to_log is None else phy_to_log.shape[0]
            new_moe_mappings = {}
            for key, mapping in moe_mappings.items():
                target_param = mapping.target_path[0]
                src_paths = mapping.target_path[1:]
                new_moe_mappings[key] = WeightMapping(
                    target_path=[target_param] + src_paths,
                    sharding=mapping.sharding,
                    transpose=mapping.transpose,
                    concat_axis=mapping.concat_axis,
                    physical_to_logical_map=mapping.physical_to_logical_map,
                )

                scale_src_paths = [src.replace(".weight", ".weight_scale_inv") for src in src_paths]
                is_w2 = target_param.endswith(("w2", "wo"))
                out_dim = hidden_size if is_w2 else inter_size
                if moe_backend == "fused":
                    in_dim = inter_size if is_w2 else hidden_size
                    scale_reshape = (num_physical_experts, 1, 1, out_dim)
                    scale_repeat = (1, in_dim // 256)
                    scale_sharding = None
                    if mapping.sharding:
                        scale_sharding = (
                            mapping.sharding[0],
                            mapping.sharding[1],
                            None,
                            mapping.sharding[2],
                        )
                else:
                    scale_reshape = (num_physical_experts, 1, 1, out_dim)
                    scale_repeat = None
                    target_dim_sharding = None
                    if mapping.sharding:
                        if is_w2 and len(mapping.sharding) > 2:
                            target_dim_sharding = mapping.sharding[2]
                        elif not is_w2 and len(mapping.sharding) > 1:
                            target_dim_sharding = mapping.sharding[1]
                    scale_sharding = (
                        (mapping.sharding[0], target_dim_sharding, None) if mapping.sharding else None
                    )
                new_moe_mappings[key + "_scale"] = WeightMapping(
                    target_path=[target_param + "_scale"] + scale_src_paths,
                    sharding=scale_sharding,
                    transpose=False,
                    reshape=scale_reshape,
                    repeat=scale_repeat,
                    concat_axis=mapping.concat_axis,
                    physical_to_logical_map=mapping.physical_to_logical_map,
                )
            mappings.update(new_moe_mappings)
        else:
            mappings.update(moe_mappings)

        num_shared = getattr(self.config, "n_shared_experts", 0) or 0
        if num_shared <= 0:
            return mappings

        if moe_backend == "fused":
            shared_map = [
                ("gate_proj", "w1_shared"),
                ("up_proj", "w3_shared"),
                ("down_proj", "w2_shared"),
            ]
            for hf_name, target_name in shared_map:
                full_hf_key = f"{prefix}.mlp.shared_experts.{hf_name}.weight"
                if is_static_quant:
                    mappings[full_hf_key] = WeightMapping(
                        target_path=f"{target_prefix}.mlp.{target_name}",
                        sharding=(None, None),
                        transpose=True,
                    )
                    out_dim = (
                        self.config.hidden_size
                        if hf_name == "down_proj"
                        else self.config.moe_intermediate_size * num_shared
                    )
                    for scale_key in _scale_source_keys(full_hf_key):
                        mappings[scale_key] = WeightMapping(
                            target_path=f"{target_prefix}.mlp.{target_name}_scale",
                            sharding=(None, None, None),
                            reshape=(1, 1, out_dim),
                            transpose=False,
                        )
                else:
                    mappings[full_hf_key] = WeightMapping(
                        target_path=f"{target_prefix}.mlp.{target_name}",
                        sharding=(None, None),
                        transpose=True,
                    )
        else:
            add_linear_mapping(
                f"{prefix}.mlp.shared_experts.gate_proj.weight",
                f"{target_prefix}.mlp.shared_experts.gate_proj",
                (None, "tensor"),
            )
            add_linear_mapping(
                f"{prefix}.mlp.shared_experts.up_proj.weight",
                f"{target_prefix}.mlp.shared_experts.up_proj",
                (None, "tensor"),
            )
            add_linear_mapping(
                f"{prefix}.mlp.shared_experts.down_proj.weight",
                f"{target_prefix}.mlp.shared_experts.down_proj",
                ("tensor", None),
            )

        return mappings

    def get_embed_and_head(self):
        return (
            self.model.embed_tokens.embedding.value,
            self.lm_head.embedding.value,
        )

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_state, layers_topk_ids = self.model(
            forward_batch,
            token_to_kv_pool,
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_state, True, layers_topk_ids


EntryClass = DeepseekV3ForCausalLM
