# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

import ast
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def test_dfly_speculative_config_routes_to_dspark_and_restores_architecture():
    from vllm_hcu.patch.platform.core_fix import patch_dfly_speculative_config

    module = ModuleType(patch_dfly_speculative_config.TARGET_MODULE)

    class SpeculativeConfig:
        @staticmethod
        def hf_config_override(hf_config):
            return hf_config

        def __post_init__(self):
            # Reproduce v0.25.1's explicit-DSpark rewrite. model_arch is a
            # checkpoint field and remains available for the HCU adapter.
            if (
                "Qwen3DSparkModel"
                not in self.draft_model_config.hf_config.architectures
            ):
                self.draft_model_config.hf_config.model_type = "deepseek_v4"
                self.draft_model_config.hf_config.architectures = [
                    "DSparkDraftModel"
                ]

        def update_arch_(self):
            self.update_arch_calls += 1

    module.SpeculativeConfig = SpeculativeConfig
    assert patch_dfly_speculative_config.apply_to_module(module)
    assert not patch_dfly_speculative_config.apply_to_module(module)

    config = SpeculativeConfig()
    config.method = "dspark"
    config.parallel_drafting = False
    config.update_arch_calls = 0
    config.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen3",
            architectures=["Qwen3DFlyModel"],
        )
    )

    config.draft_model_config.hf_config = config.hf_config_override(
        config.draft_model_config.hf_config
    )

    config.__post_init__()

    assert config.method == "dspark"
    assert config.parallel_drafting is True
    assert config.draft_model_config.hf_config.model_type == "qwen3"
    assert config.draft_model_config.hf_config.model_arch == "dfly"
    assert config.draft_model_config.hf_config.architectures == ["Qwen3DFlyModel"]
    assert config.update_arch_calls == 1

    dynamic = SpeculativeConfig()
    dynamic.method = "dspark"
    dynamic.parallel_drafting = False
    dynamic.update_arch_calls = 0
    dynamic.num_speculative_tokens_per_batch_size = [(1, 8, 4)]
    dynamic.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen3",
            architectures=["Qwen3DFlyModel"],
        )
    )
    dynamic.draft_model_config.hf_config = dynamic.hf_config_override(
        dynamic.draft_model_config.hf_config
    )
    with pytest.raises(ValueError, match="does not yet support dynamic"):
        dynamic.__post_init__()


def test_dfly_forces_hcu_v1_runner_without_affecting_other_models():
    from vllm_hcu.patch.platform.core_fix import patch_vllm_config

    module = ModuleType(patch_vllm_config.TARGET_MODULE)

    class ModelConfig:
        def get_model_arch_config(self):
            return None

    class VllmConfig:
        def with_hf_config(self, hf_config, architectures):
            return self

        def _set_cudagraph_sizes(self):
            return None

        @property
        def use_v2_model_runner(self):
            return True

    module.ModelConfig = ModelConfig
    module.VllmConfig = VllmConfig
    assert patch_vllm_config.apply_to_module(module)

    dfly = VllmConfig()
    dfly.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            architectures=["Qwen3DFlyModel"],
            hf_config=SimpleNamespace(model_arch="dfly"),
        )
    )
    assert dfly.use_v2_model_runner is False

    generic = VllmConfig()
    generic.speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            architectures=["Qwen3DSparkModel"],
            hf_config=SimpleNamespace(model_arch="dspark"),
        )
    )
    assert generic.use_v2_model_runner is True


def test_hcu_v1_dspark_proposer_rejects_non_dfly_architecture():
    from vllm_hcu.v1.spec_decode.dspark import DSparkProposer

    speculative_config = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(
            architectures=["Qwen3DSparkModel"],
            hf_config=SimpleNamespace(
                architectures=["Qwen3DSparkModel"], model_arch="dspark"
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="only Qwen3DFlyModel"):
        DSparkProposer(
            SimpleNamespace(speculative_config=speculative_config),
            torch.device("cpu"),
        )


def test_dfly_cudagraph_disables_shared_custom_all_reduce_registry():
    from vllm_hcu.patch.config import HcuFeatureConfig
    from vllm_hcu.patch.platform.core_fix import patch_vllm_config

    class CompilationConfig:
        pass

    def config(*, architecture="Qwen3DFlyModel", enforce_eager=False):
        return SimpleNamespace(
            additional_config={"hcu": HcuFeatureConfig().to_dict()},
            compilation_config=CompilationConfig(),
            model_config=SimpleNamespace(enforce_eager=enforce_eager),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                disable_custom_all_reduce=False,
            ),
            kernel_config=SimpleNamespace(moe_backend="auto"),
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(
                    architectures=[architecture],
                    hf_config=SimpleNamespace(model_arch="dfly"),
                )
            ),
        )

    graph_config = config()
    patch_vllm_config.validate_and_update_hcu_config(graph_config)
    assert graph_config.parallel_config.disable_custom_all_reduce is True

    eager_config = config(enforce_eager=True)
    patch_vllm_config.validate_and_update_hcu_config(eager_config)
    assert eager_config.parallel_config.disable_custom_all_reduce is False


def test_dfly_k_norm_uses_each_draft_layers_weight(monkeypatch):
    from vllm_hcu.models import qwen3_dfly

    calls: list[torch.Tensor] = []

    def fake_rms_norm(output, values, weight, eps):
        del eps
        calls.append(weight.clone())
        output.copy_(values * weight)

    monkeypatch.setattr(qwen3_dfly.ops, "rms_norm", fake_rms_norm)
    all_k = torch.arange(16, dtype=torch.float32).view(2, 2, 1, 4)
    weights = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    model = SimpleNamespace(_k_norm_weights=weights, _rms_norm_eps=1e-6)

    output = qwen3_dfly.Qwen3DFlyModel._normalize_context_k(model, all_k)

    assert len(calls) == 2
    torch.testing.assert_close(output[0], all_k[0] * weights[0])
    torch.testing.assert_close(output[1], all_k[1] * weights[1])


def test_dfly_context_kv_buffer_accepts_hcu_nn_and_standard_layouts():
    from vllm_hcu.models import qwen3_dfly

    # Logical rows are Q0, Q1, K, V.  Exercise one conventional [out, in]
    # layer and one HCU NN-layout [in, out] layer in the same derived buffer.
    logical_a = torch.arange(12, dtype=torch.float32).view(4, 3)
    logical_b = logical_a + 100

    def attention(weight):
        return SimpleNamespace(
            q_size=2,
            kv_size=1,
            qkv_proj=SimpleNamespace(weight=torch.nn.Parameter(weight), bias=None),
            k_norm=SimpleNamespace(
                weight=torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            ),
        )

    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=3),
        hidden_norm=SimpleNamespace(
            weight=torch.nn.Parameter(torch.ones(3))
        ),
    )
    qwen3_dfly.Qwen3DFlyModel._build_context_kv_buffers(
        model,
        [attention(logical_a), attention(logical_b.t().contiguous())],
        has_bias=False,
    )

    torch.testing.assert_close(
        model._fused_kv_weight,
        torch.cat([logical_a[2:], logical_b[2:]], dim=0),
    )
    assert model._fused_kv_weight.shape == (4, 3)
    assert model._fused_kv_bias is None


def test_dspark_dfly_sampling_feeds_each_sample_to_the_next_step():
    from vllm_hcu.v1.spec_decode.dspark import DSparkProposer

    class FakeDFlyModel:
        @staticmethod
        def has_hidden_correction():
            return True

        @staticmethod
        def has_markov():
            return False

        @staticmethod
        def apply_hidden_correction(hidden_states, prev_token_ids):
            return prev_token_ids.to(hidden_states.dtype).unsqueeze(-1)

        @staticmethod
        def compute_draft_logits(hidden_states):
            token_ids = (hidden_states[:, 0].long() + 1) % 8
            logits = torch.full((hidden_states.shape[0], 8), -100.0)
            return logits.scatter_(1, token_ids.unsqueeze(1), 100.0)

        @staticmethod
        def map_draft_to_target(draft_ids):
            return draft_ids

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.model = FakeDFlyModel()
    proposer.num_speculative_tokens = 2
    proposer.device = torch.device("cpu")
    proposer.input_ids = torch.tensor([2, 0, 5, 0], dtype=torch.int32)
    proposer._anchor_idx = torch.tensor([0, 2], dtype=torch.int64)
    proposer._enable_probabilistic_draft_probs = False
    proposer.markov_position_offset = 0

    token_ids, probs = proposer._sample_draft_tokens(
        torch.zeros((4, 1)),
        SimpleNamespace(all_greedy=True),
    )

    assert probs is None
    assert token_ids.tolist() == [3, 4, 6, 7]


def test_dspark_resolves_dfly_dflare_mask_token_alias():
    from vllm_hcu.v1.spec_decode.dspark import DSparkProposer

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(dflare_config={"mask_token_id": 120831})
    )
    proposer.pass_hidden_states_to_model = False

    proposer._init_parallel_drafting_params()

    assert proposer.parallel_drafting_token_id == 120831
    assert proposer.dflash_config == {"mask_token_id": 120831}


def test_dspark_dummy_run_reuses_persistent_context_buffer(monkeypatch):
    from vllm.config import CUDAGraphMode
    from vllm_hcu.v1.spec_decode import dspark

    context_inputs = []

    class FakeModel:
        def precompute_and_store_context_kv(self, states, positions):
            context_inputs.append(states)

        def __call__(self, **kwargs):
            return torch.zeros((kwargs["input_ids"].shape[0], 1))

    monkeypatch.setattr(dspark, "set_forward_context", lambda *a, **k: nullcontext())
    proposer = dspark.DSparkProposer.__new__(dspark.DSparkProposer)
    proposer.max_query_tokens = 8
    proposer._draft_attn_layer_names = []
    proposer.model = FakeModel()
    proposer._dummy_context_hidden_states = torch.zeros((16, 6))
    proposer._context_positions_buffer = torch.zeros(16, dtype=torch.int64)
    proposer.input_ids = torch.zeros(8, dtype=torch.int32)
    proposer.positions = torch.zeros(8, dtype=torch.int64)
    proposer.vllm_config = SimpleNamespace()
    proposer._determine_batch_execution_and_padding = lambda n, **k: (
        CUDAGraphMode.PIECEWISE,
        n,
        None,
    )
    proposer._get_positions = lambda n: proposer.positions[:n]

    proposer.dummy_run(4)
    proposer.dummy_run(4)

    assert len(context_inputs) == 2
    assert context_inputs[0].data_ptr() == context_inputs[1].data_ptr()
    assert (
        context_inputs[0].data_ptr()
        == proposer._dummy_context_hidden_states.data_ptr()
    )

    profile_calls = []
    proposer._dummy_sample_run = (
        lambda hidden_states, num_reqs: profile_calls.append(
            (hidden_states.shape[0], num_reqs)
        )
    )
    proposer.dummy_run(4, profile_num_reqs=1)
    assert profile_calls == [(4, 1)]


def test_hy3_dfly_aux_layer_ids_use_decoder_boundaries():
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu"
        / "v1"
        / "hcu_model_runner.py"
    ).read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_get_eagle3_aux_layers_from_config"
    )
    method_source = ast.get_source_segment(source, method)
    assert method_source is not None
    assert 'getattr(hf_config, "dflash_config", None)' in method_source
    assert 'getattr(hf_config, "dflare_config", None)' in method_source
    assert "[i + 1 for i in target_layer_ids]" in method_source


def test_dfly_missing_required_checkpoint_parameters_are_fatal():
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_hcu"
        / "models"
        / "qwen3_dfly.py"
    ).read_text()
    assert "DFly checkpoint is missing required parameters" in source
    assert "loaded_param_names.update(" in source
