import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import save_file
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import AttentionArch, ModelConfig
from sgl_jax.srt.layers.attention.flashattention_backend import FlashAttention
from sgl_jax.srt.layers.attention.native_backend import NativeAttention
from sgl_jax.srt.layers.logits_processor import LogitsMetadata
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.models.deepseek_v3 import (
    DeepseekV3ForCausalLM,
    DeepseekV3MoE,
)
from sgl_jax.srt.models.registry import ModelRegistry
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


def tiny_deepseek_config_dict(**overrides):
    config = {
        "model_type": "llama",
        "architectures": ["DeepseekV3ForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "q_lora_rank": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "ep_size": 1,
        "n_group": 2,
        "topk_group": 1,
        "routed_scaling_factor": 1.5,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "first_k_dense_replace": 0,
        "moe_layer_freq": 1,
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    config.update(overrides)
    return config


def tiny_pretrained_config(**overrides):
    config = PretrainedConfig()
    for key, value in tiny_deepseek_config_dict(**overrides).items():
        setattr(config, key, value)
    return config


def make_decode_batches(attn_backend, seq_len: int):
    input_ids = np.array([1], dtype=np.int32)
    seq_lens = np.array([seq_len], dtype=np.int32)
    out_cache_loc = np.array([seq_len - 1], dtype=np.int32)
    cache_loc = np.arange(seq_len, dtype=np.int32)
    positions = np.array([seq_len - 1], dtype=np.int32)
    req_pool_indices = np.array([0], dtype=np.int32)

    mwb = ModelWorkerBatch(
        bid=1,
        forward_mode=ForwardMode.DECODE,
        input_ids=input_ids,
        real_input_ids_len=1,
        seq_lens=seq_lens,
        out_cache_loc=out_cache_loc,
        req_pool_indices=req_pool_indices,
        sampling_info=None,
        positions=positions,
        cache_loc=cache_loc,
        extend_seq_lens=None,
        extend_prefix_lens=None,
        return_logprob=False,
        return_output_logprob_only=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        extend_logprob_start_lens=None,
        extend_input_logprob_token_ids=None,
        real_bs=1,
        spec_info=None,
    )
    fb = ForwardBatch(
        bid=1,
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        input_ids=jnp.array(input_ids),
        req_pool_indices=jnp.array(req_pool_indices),
        seq_lens=jnp.array(seq_lens),
        out_cache_loc=jnp.array(out_cache_loc),
        positions=jnp.array(positions),
        attn_backend=attn_backend,
        cache_loc=jnp.array(cache_loc),
        extend_prefix_lens=None,
        extend_seq_lens=None,
        spec_info=None,
    )
    metadata = attn_backend.get_forward_metadata(mwb)
    if metadata is not None:
        fb.attn_backend.forward_metadata = metadata
    return mwb, fb


def dense_checkpoint_tensors(config: dict[str, int | float | bool | str | None]):
    hidden = config["hidden_size"]
    vocab = config["vocab_size"]
    inter = config["intermediate_size"]
    q_rank = config["q_lora_rank"]
    kv_rank = config["kv_lora_rank"]
    q_head_dim = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]
    num_heads = config["num_attention_heads"]
    v_head_dim = config["v_head_dim"]

    cursor = 1

    def tensor(shape):
        nonlocal cursor
        size = int(np.prod(shape))
        value = np.arange(cursor, cursor + size, dtype=np.float32).reshape(shape)
        cursor += size
        return value

    return {
        "model.embed_tokens.weight": tensor((vocab, hidden)),
        "model.norm.weight": tensor((hidden,)),
        "lm_head.weight": tensor((vocab, hidden)),
        "model.layers.0.input_layernorm.weight": tensor((hidden,)),
        "model.layers.0.post_attention_layernorm.weight": tensor((hidden,)),
        "model.layers.0.self_attn.q_a_proj.weight": tensor((q_rank, hidden)),
        "model.layers.0.self_attn.q_a_layernorm.weight": tensor((q_rank,)),
        "model.layers.0.self_attn.q_b_proj.weight": tensor((num_heads * q_head_dim, q_rank)),
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight": tensor(
            (kv_rank + config["qk_rope_head_dim"], hidden)
        ),
        "model.layers.0.self_attn.kv_a_layernorm.weight": tensor((kv_rank,)),
        "model.layers.0.self_attn.kv_b_proj.weight": tensor(
            (num_heads * (config["qk_nope_head_dim"] + v_head_dim), kv_rank)
        ),
        "model.layers.0.self_attn.o_proj.weight": tensor((hidden, num_heads * v_head_dim)),
        "model.layers.0.mlp.gate_proj.weight": tensor((inter, hidden)),
        "model.layers.0.mlp.up_proj.weight": tensor((inter, hidden)),
        "model.layers.0.mlp.down_proj.weight": tensor((hidden, inter)),
    }


class TestDeepseekV3(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def test_model_config_detects_mla_and_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(tiny_deepseek_config_dict()), encoding="utf-8")

            model_config = ModelConfig(tmpdir, trust_remote_code=False)

            self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
            self.assertEqual(model_config.head_dim, 8)
            self.assertEqual(model_config.v_head_dim, 4)
            self.assertEqual(model_config.kv_lora_rank, 4)
            self.assertEqual(model_config.q_lora_rank, 8)
            self.assertEqual(model_config.num_experts, 4)
            self.assertEqual(model_config.hf_config.moe_backend, "fused")
            model_cls, arch = ModelRegistry.resolve_model_cls(model_config.hf_config.architectures)
            self.assertEqual(arch, "DeepseekV3ForCausalLM")
            self.assertEqual(model_cls.__name__, "DeepseekV3ForCausalLM")

    def test_mla_kv_pool_reconstructs_full_k(self):
        pool = MLATokenToKVPool(
            size=8,
            page_size=1,
            dtype=jnp.bfloat16,
            head_num=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            layer_num=1,
            mesh=mesh,
        )
        loc = jnp.array([1, 3], dtype=jnp.int32)
        k_nope = jnp.arange(2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4).astype(jnp.bfloat16)
        k_pe = jnp.arange(2 * 1 * 4, dtype=jnp.float32).reshape(2, 1, 4).astype(jnp.bfloat16)
        v = (k_nope + 100).astype(jnp.bfloat16)

        pool.set_mla_kv_buffer(0, loc, k_nope, k_pe, v)

        stored_k_nope, stored_k_pe, stored_v = pool.get_mla_kv_buffer(0)
        np.testing.assert_allclose(np.asarray(stored_k_nope[loc]), np.asarray(k_nope))
        np.testing.assert_allclose(np.asarray(stored_k_pe[loc]), np.asarray(k_pe))
        np.testing.assert_allclose(np.asarray(stored_v[loc]), np.asarray(v))

        full_k, full_v = pool.get_split_kv_buffer(0)
        expected_full_k = jnp.concatenate(
            [k_nope, jnp.broadcast_to(k_pe, k_nope.shape[:-1] + (k_pe.shape[-1],))],
            axis=-1,
        )
        np.testing.assert_allclose(np.asarray(full_k[loc]), np.asarray(expected_full_k))
        np.testing.assert_allclose(np.asarray(full_v[loc]), np.asarray(v))

    def test_flash_mla_backend_matches_reference_kernel(self):
        backend = FlashAttention(
            num_attn_heads=2,
            num_kv_heads=2,
            head_dim=8,
            page_size=1,
            mesh=mesh,
            v_head_dim=4,
        )
        _, forward_batch = make_decode_batches(backend, seq_len=3)
        pool = MLATokenToKVPool(
            size=8,
            page_size=1,
            dtype=jnp.bfloat16,
            head_num=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            layer_num=1,
            mesh=mesh,
        )

        prefix_k_nope = jnp.array(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]],
            ],
            dtype=jnp.bfloat16,
        )
        prefix_k_pe = jnp.array(
            [[[0.2, 0.0, 0.0, 0.0]], [[0.0, 0.2, 0.0, 0.0]]],
            dtype=jnp.bfloat16,
        )
        prefix_v = jnp.array(
            [
                [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
                [[2.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 2.0]],
            ],
            dtype=jnp.bfloat16,
        )
        pool.k_nope_buffer[0] = pool.k_nope_buffer[0].at[jnp.array([0, 1])].set(prefix_k_nope)
        pool.k_pe_buffer[0] = pool.k_pe_buffer[0].at[jnp.array([0, 1])].set(prefix_k_pe)
        pool.v_buffer[0] = pool.v_buffer[0].at[jnp.array([0, 1])].set(prefix_v)

        q = jnp.array(
            [[[0.3, 0.4, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0], [0.4, 0.3, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]]],
            dtype=jnp.bfloat16,
        )
        k_nope = jnp.array(
            [[[0.4, 0.2, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0]]],
            dtype=jnp.bfloat16,
        )
        k_pe = jnp.array([[[0.1, 0.1, 0.0, 0.0]]], dtype=jnp.bfloat16)
        v = jnp.array(
            [[[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]]],
            dtype=jnp.bfloat16,
        )

        layer = RadixAttention(
            num_heads=2,
            head_dim=8,
            scaling=8**-0.5,
            num_kv_heads=2,
            layer_id=0,
            v_head_dim=4,
        )

        expected_k = jnp.concatenate(
            [
                jnp.concatenate(
                    [
                        prefix_k_nope,
                        jnp.broadcast_to(
                            prefix_k_pe,
                            prefix_k_nope.shape[:-1] + (prefix_k_pe.shape[-1],),
                        ),
                    ],
                    axis=-1,
                ),
                jnp.concatenate(
                    [k_nope, jnp.broadcast_to(k_pe, k_nope.shape[:-1] + (k_pe.shape[-1],))],
                    axis=-1,
                ),
            ],
            axis=0,
        )
        expected_v = jnp.concatenate([prefix_v, v], axis=0)
        scores = jnp.einsum("qhd,khd->hqk", q, expected_k, preferred_element_type=jnp.float32)
        scores *= layer.scaling
        ref_out = jnp.einsum(
            "hqk,khd->qhd",
            jax.nn.softmax(scores, axis=-1).astype(expected_v.dtype),
            expected_v,
        )

        def fake_ragged_paged_attention(
            queries,
            keys_new,
            values_new,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            custom_mask,
            *,
            k_cache=None,
            v_cache=None,
            sm_scale=1.0,
            **kwargs,
        ):
            del kv_cache_fused, kv_lens, cu_q_lens, cu_kv_lens, distribution, custom_mask, kwargs
            k_flat = k_cache.reshape(-1, k_cache.shape[2], k_cache.shape[3])
            v_flat = v_cache.reshape(-1, v_cache.shape[2], v_cache.shape[3])
            seq_page_indices = page_indices[0] if page_indices.ndim == 2 else page_indices
            write_idx = seq_page_indices[-keys_new.shape[0] :]
            k_flat = k_flat.at[write_idx].set(keys_new)
            v_flat = v_flat.at[write_idx].set(values_new)
            k_seq = k_flat[seq_page_indices]
            v_seq = v_flat[seq_page_indices]
            attn_scores = jnp.einsum(
                "qhd,khd->hqk",
                queries,
                k_seq,
                preferred_element_type=jnp.float32,
            )
            attn_scores *= sm_scale
            attn_weights = jax.nn.softmax(attn_scores, axis=-1).astype(v_seq.dtype)
            attn_out = jnp.einsum("hqk,khd->qhd", attn_weights, v_seq)
            return attn_out, k_flat, v_flat

        with mock.patch(
            "sgl_jax.srt.layers.attention.flashattention_backend.ragged_paged_attention",
            new=fake_ragged_paged_attention,
        ):
            attn_output, kv_state = backend(
                q,
                k_nope,
                v,
                layer,
                forward_batch,
                pool,
                k_pe=k_pe,
            )

        np.testing.assert_allclose(
            np.asarray(attn_output).astype(np.float32),
            np.asarray(ref_out.reshape(1, -1)).astype(np.float32),
            rtol=1e-4,
            atol=1e-4,
        )
        updated_k_nope, updated_k_pe, updated_v = kv_state
        np.testing.assert_allclose(
            np.asarray(updated_k_nope[2]).astype(np.float32),
            np.asarray(k_nope[0]).astype(np.float32),
        )
        np.testing.assert_allclose(
            np.asarray(updated_k_pe[2]).astype(np.float32),
            np.asarray(k_pe[0]).astype(np.float32),
        )
        np.testing.assert_allclose(
            np.asarray(updated_v[2]).astype(np.float32),
            np.asarray(v[0]).astype(np.float32),
        )

    def test_deepseek_moe_uses_sigmoid_grouped_topk_and_shared_experts(self):
        config = tiny_pretrained_config(
            moe_backend="epmoe",
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            n_group=2,
            topk_group=1,
            routed_scaling_factor=1.5,
        )
        with jax.set_mesh(mesh):
            moe = DeepseekV3MoE(config, mesh=mesh, layer_id=0)
            hidden_states = jnp.ones((3, config.hidden_size), dtype=jnp.bfloat16)
            router_logits = moe.moe_gate(hidden_states)
            topk_weights, topk_ids = moe.topk(
                router_logits,
                correction_bias=moe.moe_gate.bias.value,
            )

        self.assertEqual(moe.moe_gate.score_func, "sigmoid")
        self.assertIsNotNone(moe.moe_gate.bias)
        self.assertTrue(moe.topk.renormalize)
        self.assertEqual(moe.topk.num_expert_group, 2)
        self.assertIsNotNone(moe.shared_experts)
        self.assertEqual(topk_ids.shape, (3, 2))
        np.testing.assert_allclose(
            np.asarray(topk_weights.sum(axis=-1)),
            np.full((3,), 1.5, dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_tiny_checkpoint_loads_into_native_deepseek_model(self):
        config_dict = tiny_deepseek_config_dict(
            n_routed_experts=None,
            n_shared_experts=0,
            first_k_dense_replace=99,
        )
        tensors = dense_checkpoint_tensors(config_dict)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
            save_file(tensors, str(tmp_path / "model.safetensors"))

            model_config = ModelConfig(tmpdir, trust_remote_code=False)
            with jax.set_mesh(mesh):
                model = DeepseekV3ForCausalLM(model_config.hf_config, mesh=mesh)
                model.load_weights(model_config)

            loaded_q_a = np.asarray(model.model.layers[0].self_attn.q_a_proj.weight.value).astype(
                np.float32
            )
            expected_q_a = (
                tensors["model.layers.0.self_attn.q_a_proj.weight"].T.astype(jnp.bfloat16)
            ).astype(np.float32)
            np.testing.assert_allclose(loaded_q_a, expected_q_a)

            loaded_down = np.asarray(model.model.layers[0].mlp.down_proj.weight.value).astype(
                np.float32
            )
            expected_down = (
                tensors["model.layers.0.mlp.down_proj.weight"].T.astype(jnp.bfloat16)
            ).astype(np.float32)
            np.testing.assert_allclose(loaded_down, expected_down)

    def test_deepseek_dense_decode_native_smoke(self):
        config = tiny_pretrained_config(
            moe_backend="epmoe",
            n_routed_experts=None,
            n_shared_experts=0,
            first_k_dense_replace=99,
        )
        backend = NativeAttention(num_attn_heads=2, num_kv_heads=2, mesh=mesh)
        _, forward_batch = make_decode_batches(backend, seq_len=1)
        token_to_kv_pool = MLATokenToKVPool(
            size=8,
            page_size=1,
            dtype=jnp.bfloat16,
            head_num=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            layer_num=1,
            mesh=mesh,
        )
        logits_metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)

        with jax.set_mesh(mesh):
            model = DeepseekV3ForCausalLM(config, mesh=mesh)
            output, layers_kv, need_sample, topk_ids = model(
                forward_batch,
                token_to_kv_pool,
                logits_metadata,
            )

        self.assertEqual(output.next_token_logits.shape, (1, config.vocab_size))
        self.assertTrue(need_sample)
        self.assertEqual(len(layers_kv), 1)
        self.assertIsNone(topk_ids[0])
        self.assertEqual(layers_kv[0][0].shape[-1], config.qk_nope_head_dim)
        self.assertEqual(layers_kv[0][1].shape[-1], config.qk_rope_head_dim)
        self.assertEqual(layers_kv[0][2].shape[-1], config.v_head_dim)


if __name__ == "__main__":
    unittest.main()
