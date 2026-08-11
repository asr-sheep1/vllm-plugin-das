# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

from __future__ import annotations

import hashlib
import msgspec
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch


def test_dcut_config_accepts_dfly_and_rejects_invalid_values():
    from vllm_hcu.patch.platform.core_fix import patch_dcut_speculative_config

    module = ModuleType(patch_dcut_speculative_config.TARGET_MODULE)
    module.safe_hash = hashlib.sha256
    module.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)

    class SpeculativeConfig:
        def __init__(self, method="dspark", draft_model_config=None):
            self.method = method
            self.draft_model_config = draft_model_config

        def compute_hash(self):
            return "upstream"

    module.SpeculativeConfig = SpeculativeConfig
    assert patch_dcut_speculative_config.apply_to_module(module)
    assert not patch_dcut_speculative_config.apply_to_module(module)

    dfly_model = SimpleNamespace(
        architectures=["Qwen3DFlyModel"],
        hf_config=SimpleNamespace(
            architectures=["Qwen3DFlyModel"], model_arch="dfly"
        ),
    )
    config = SpeculativeConfig(
        method="dspark",
        draft_model_config=dfly_model,
        dflash_dcut=0.5,
    )
    assert config.dflash_dcut == 0.5
    assert config.dflash_dcut_mode == "fixed_ratio"
    assert config.uses_dflash_dcut()
    assert config.compute_hash() != "upstream"

    selector = SpeculativeConfig(
        method="dspark",
        draft_model_config=dfly_model,
        dflash_dcut="auto",
    )
    assert selector.dflash_dcut_mode == "selector"

    generic_dspark = SpeculativeConfig(
        method="dspark",
        draft_model_config=SimpleNamespace(
            architectures=["Qwen3DSparkModel"],
            hf_config=SimpleNamespace(architectures=["Qwen3DSparkModel"]),
        ),
        dflash_dcut=0.5,
    )
    assert generic_dspark.dflash_dcut == 0.0
    assert not generic_dspark.uses_dflash_dcut()

    generic_dflash = SpeculativeConfig(
        method="dflash",
        draft_model_config=SimpleNamespace(
            architectures=["DFlashDraftModel"],
            hf_config=SimpleNamespace(architectures=["DFlashDraftModel"]),
        ),
        dflash_dcut=0.5,
    )
    assert generic_dflash.dflash_dcut == 0.0
    assert not generic_dflash.uses_dflash_dcut()

    with pytest.raises(ValueError, match="dflash_dcut"):
        SpeculativeConfig(
            method="dspark",
            draft_model_config=dfly_model,
            dflash_dcut=1.1,
        )


def test_dcut_config_hash_includes_dfly_target_layers():
    from vllm_hcu.patch.platform.core_fix import patch_dcut_speculative_config

    module = ModuleType(patch_dcut_speculative_config.TARGET_MODULE)
    module.safe_hash = hashlib.sha256
    module.logger = SimpleNamespace(warning=lambda *args, **kwargs: None)

    class DraftModelConfig:
        def __init__(self, target_layer_ids):
            self.architectures = ["Qwen3DFlyModel"]
            self.hf_config = SimpleNamespace(
                architectures=["Qwen3DFlyModel"],
                model_arch="dfly",
                target_layer_ids=target_layer_ids,
            )

        def compute_hash(self):
            return "same-draft-model-hash"

    class SpeculativeConfig:
        def __init__(
            self,
            draft_model_config,
            num_speculative_tokens=7,
            **kwargs,
        ):
            self.method = "dspark"
            self.draft_model_config = draft_model_config
            self.num_speculative_tokens = num_speculative_tokens
            self.draft_sample_method = "greedy"
            self.attention_backend = None

        def compute_hash(self):
            return "same-upstream-hash"

    module.SpeculativeConfig = SpeculativeConfig
    assert patch_dcut_speculative_config.apply_to_module(module)

    first = SpeculativeConfig(DraftModelConfig([1, 20, 39]), dflash_dcut=0.0)
    second = SpeculativeConfig(DraftModelConfig([2, 21, 40]), dflash_dcut=0.0)

    assert first.compute_hash() != second.compute_hash()
    different_k = SpeculativeConfig(
        DraftModelConfig([1, 20, 39]),
        num_speculative_tokens=4,
        dflash_dcut=0.0,
    )
    assert first.compute_hash() != different_k.compute_hash()


def test_dcut_draft_token_transport_is_optional_and_copy_safe():
    import copy

    from vllm_hcu.patch.platform.framework_opt import patch_outputs

    module = ModuleType(patch_outputs.TARGET_MODULE)

    @dataclass
    class ModelRunnerOutput:
        req_ids: list[str]

    @dataclass
    class DraftTokenIds:
        req_ids: list[str]
        draft_token_ids: list[list[int]]

    module.ModelRunnerOutput = ModelRunnerOutput
    module.DraftTokenIds = DraftTokenIds
    module.EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput([])
    assert patch_outputs.apply_to_module(module)

    legacy = DraftTokenIds(["r0"], [[1, 2]])
    assert legacy.dcut_keep_lens is None
    dcut = DraftTokenIds(["r0", "r1"], [[1, 2], [3, 4]], [1, 0])
    assert dcut.dcut_keep_lens == [1, 0]
    restored = copy.deepcopy(dcut)
    assert restored.dcut_keep_lens == [1, 0]
    payload = msgspec.msgpack.encode(dcut)
    decoded = msgspec.msgpack.Decoder(DraftTokenIds).decode(payload)
    assert decoded.dcut_keep_lens == [1, 0]


def test_dcut_metrics_accumulate_and_log_keep_ratio():
    from vllm_hcu.patch.platform.framework_opt import patch_dcut_metrics

    module = ModuleType(patch_dcut_metrics.TARGET_MODULE)
    module.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    @dataclass
    class SpecDecodingStats:
        num_drafts: int = 1

    class SpecDecodingLogging:
        def __init__(self):
            self.reset()

        def reset(self):
            self.num_drafts = []

        def observe(self, stats):
            self.num_drafts.append(stats.num_drafts)

        def log(self, log_fn):
            if self.num_drafts:
                log_fn("base")
            self.reset()

    module.SpecDecodingStats = SpecDecodingStats
    module.SpecDecodingLogging = SpecDecodingLogging
    assert patch_dcut_metrics.apply_to_module(module)

    stats = SpecDecodingStats()
    stats.observe_dcut(kept_draft_tokens=3, total_draft_tokens=8)
    stats.observe_dcut(kept_draft_tokens=2, total_draft_tokens=4)
    assert stats.dcut_kept_draft_tokens == 5
    assert stats.dcut_total_draft_tokens == 12

    payload = msgspec.msgpack.encode(stats)
    restored = msgspec.msgpack.Decoder(SpecDecodingStats).decode(payload)
    assert restored.dcut_kept_draft_tokens == 5
    assert restored.dcut_total_draft_tokens == 12

    logging = SpecDecodingLogging()
    logging.observe(stats)
    messages: list[str] = []

    def capture(message, *args):
        messages.append(message % args if args else message)

    logging.log(capture)
    assert any("keep ratio: 0.417" in message for message in messages)


def _fake_scheduler_module():
    from vllm_hcu.patch.platform.framework_opt import patch_scheduler

    class Scheduler:
        def schedule(self, throttle_prefills=False):
            return None

        def update_draft_token_ids(self, draft_token_ids):
            for req_id, token_ids in zip(
                draft_token_ids.req_ids, draft_token_ids.draft_token_ids
            ):
                request = self.requests.get(req_id)
                if request is not None:
                    request.spec_token_ids = token_ids

        def update_draft_token_ids_in_output(self, draft_token_ids, scheduler_output):
            return None

        def make_stats(
            self,
            spec_decoding_stats=None,
            kv_connector_stats=None,
            cudagraph_stats=None,
            perf_stats=None,
        ):
            return spec_decoding_stats

    for name in (
        "_select_waiting_queue_for_scheduling",
        "_is_blocked_waiting_status",
        "_try_promote_blocked_waiting_request",
        "_try_schedule_encoder_inputs",
        "_mamba_block_aligned_split",
        "_build_kv_connector_meta",
        "_inflight_prefill_reserved_blocks",
        "_make_cached_request_data",
        "_update_after_schedule",
        "_preempt_request",
    ):
        setattr(Scheduler, name, lambda self, *args, **kwargs: None)
    return ModuleType(patch_scheduler.TARGET_MODULE), Scheduler


def test_dcut_scheduler_truncates_prefix_and_drains_accounting():
    from vllm_hcu.patch.platform.framework_opt import patch_scheduler

    module, scheduler_class = _fake_scheduler_module()
    module.Scheduler = scheduler_class
    assert patch_scheduler.apply_to_module(module)

    request0 = SimpleNamespace(
        spec_token_ids=[],
        is_prefill_chunk=False,
        is_finished=lambda: False,
    )
    request1 = SimpleNamespace(
        spec_token_ids=[],
        is_prefill_chunk=False,
        is_finished=lambda: False,
    )
    scheduler = scheduler_class()
    scheduler.requests = {"r0": request0, "r1": request1}
    scheduler.log_stats = True
    draft = SimpleNamespace(
        req_ids=["r0", "r1"],
        draft_token_ids=[[10, 11, 12], [20, 21, 22]],
        dcut_keep_lens=[2, 0],
    )
    scheduler.update_draft_token_ids(draft)
    assert request0.spec_token_ids == [10, 11]
    assert request1.spec_token_ids == []
    assert scheduler._pending_dcut_kept_draft_tokens == 4
    assert scheduler._pending_dcut_total_draft_tokens == 8

    stats = SimpleNamespace(
        dcut_kept_draft_tokens=0,
        dcut_total_draft_tokens=0,
    )

    def observe_dcut(*, kept_draft_tokens, total_draft_tokens):
        stats.dcut_kept_draft_tokens += kept_draft_tokens
        stats.dcut_total_draft_tokens += total_draft_tokens

    stats.observe_dcut = observe_dcut
    assert scheduler.make_stats(stats) is stats
    assert stats.dcut_kept_draft_tokens == 4
    assert stats.dcut_total_draft_tokens == 8
    assert scheduler._pending_dcut_total_draft_tokens == 0


def test_dfly_dcut_fixed_ratio_uses_global_prefix_topk():
    from vllm_hcu.v1.spec_decode.dspark import DSparkProposer

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.device = torch.device("cpu")
    proposer.speculative_config = SimpleNamespace(
        dflash_dcut=0.5,
        dflash_dcut_mode="fixed_ratio",
    )
    proposer._dcut_costs_by_bs = {}
    proposer._dcut_keep_counts = None
    logprobs = torch.tensor(
        [[-0.10, -0.10, -0.10], [-0.01, -5.00, -5.00]],
        dtype=torch.float32,
    )
    keep_lens = proposer._select_dcut_keep_lens(logprobs)
    assert keep_lens.tolist() == [1, 1]
    assert DSparkProposer._get_dcut_keep_count(2, 3, 0.5) == 2


def test_dfly_dcut_auto_counts_bonus_tokens_in_throughput_score():
    from vllm_hcu.v1.spec_decode.dspark import DSparkProposer

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.device = torch.device("cpu")
    proposer.speculative_config = SimpleNamespace(
        dflash_dcut="auto",
        dflash_dcut_mode="selector",
    )
    proposer._dcut_costs_by_bs = {
        2: torch.tensor([1.0, 2.0, 3.0, 4.0]),
    }
    proposer._dcut_keep_counts = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 4, 6]],
        dtype=torch.long,
    )

    keep_lens = proposer._select_dcut_keep_lens(torch.zeros((2, 3)))

    # Including the two guaranteed bonus tokens makes all candidates tie at
    # two output tokens/ms, so argmax selects the narrowest candidate.
    assert keep_lens.tolist() == [0, 0]


def test_dfly_dcut_logprobs_keep_fp32_resolution():
    from vllm_hcu.v1.spec_decode.dspark import _token_logprobs_from_logits

    logits = torch.tensor(
        [[12.0, 9.5, 7.0], [20.0, 19.875, 10.0]],
        dtype=torch.bfloat16,
    )
    token_ids = torch.tensor([0, 1])

    actual = _token_logprobs_from_logits(logits, token_ids)
    expected = torch.log_softmax(logits.double(), dim=-1).gather(
        1, token_ids.unsqueeze(1)
    ).squeeze(1)
    torch.testing.assert_close(actual.double(), expected, atol=5e-4, rtol=0)


def test_dfly_dcut_auto_profile_builds_all_ratio_costs():
    from vllm_hcu.v1.spec_decode.dspark import DSparkProposer

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.device = torch.device("cpu")
    proposer.max_batch_size = 4
    proposer.num_speculative_tokens = 7
    proposer.num_query_per_req = 7
    proposer._runner = SimpleNamespace(max_num_reqs=4, max_num_tokens=32)
    proposer.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[8, 16, 32]
    )
    proposer._dcut_costs_by_bs = {}
    profile_calls = []

    def profile(*, batch_size, target_tokens, draft_tokens):
        profile_calls.append((batch_size, target_tokens, draft_tokens))
        return float(target_tokens + draft_tokens / 1000 + batch_size / 10000)

    proposer._profile_dcut_full_cost_ms = profile
    proposer.profile_dcut_cost_table()

    assert tuple(proposer._dcut_costs_by_bs) == (1, 2, 4)
    assert all(costs.shape == (4,) for costs in proposer._dcut_costs_by_bs.values())
    assert proposer._dcut_keep_counts.shape == (5, 4)
    assert {draft_tokens for _, _, draft_tokens in profile_calls} == {7, 14, 28}


def test_dfly_dcut_auto_profile_caps_batch_and_clamps_noisy_costs():
    from vllm_hcu.v1.spec_decode.dspark import DSparkProposer

    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.device = torch.device("cpu")
    proposer.max_batch_size = 8
    proposer.num_speculative_tokens = 7
    proposer.num_query_per_req = 8
    proposer._runner = SimpleNamespace(max_num_reqs=8, max_num_tokens=16)
    proposer.compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[8, 16, 32, 64]
    )
    proposer._dcut_costs_by_bs = {}
    raw_costs = iter([4.0, 3.0, 2.0, 1.0] * 2)
    proposer._profile_dcut_full_cost_ms = lambda **kwargs: next(raw_costs)

    proposer.profile_dcut_cost_table()

    # A full verification step uses eight target tokens per request, so the
    # 16-token runner limit permits at most two requests during profiling.
    assert tuple(proposer._dcut_costs_by_bs) == (1, 2)
    for costs in proposer._dcut_costs_by_bs.values():
        assert torch.all(costs[1:] >= costs[:-1])
