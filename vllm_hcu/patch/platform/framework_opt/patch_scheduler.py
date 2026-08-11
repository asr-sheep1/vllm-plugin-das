# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Select and validate the HCU Scheduler through ``scheduler_cls``."""

from __future__ import annotations

import functools
from types import ModuleType

from vllm_hcu.patch.config import get_hcu_config
from vllm_hcu.platforms import envs as henvs

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_signature_prefix,
)

TARGET_MODULE = "vllm.v1.core.sched.scheduler"
PATCH_ID = "platform.framework_opt.hcu_scheduler"
TARGETS = (
    f"{TARGET_MODULE}.Scheduler",
    "vllm_hcu.v1.core.sched.scheduler.HcuScheduler",
    f"{TARGET_MODULE}.Scheduler.update_draft_token_ids",
    f"{TARGET_MODULE}.Scheduler.schedule",
    f"{TARGET_MODULE}.Scheduler.update_draft_token_ids_in_output",
    f"{TARGET_MODULE}.Scheduler.make_stats",
    f"{TARGET_MODULE}.Scheduler._select_waiting_queue_for_scheduling",
    f"{TARGET_MODULE}.Scheduler._is_blocked_waiting_status",
    f"{TARGET_MODULE}.Scheduler._try_promote_blocked_waiting_request",
    f"{TARGET_MODULE}.Scheduler._try_schedule_encoder_inputs",
    f"{TARGET_MODULE}.Scheduler._mamba_block_aligned_split",
    f"{TARGET_MODULE}.Scheduler._build_kv_connector_meta",
    f"{TARGET_MODULE}.Scheduler._inflight_prefill_reserved_blocks",
    f"{TARGET_MODULE}.Scheduler._make_cached_request_data",
    f"{TARGET_MODULE}.Scheduler._update_after_schedule",
    f"{TARGET_MODULE}.Scheduler._preempt_request",
)
_MARKER = "_vllm_hcu_scheduler_contract_validated"
HCU_SCHEDULER_PATH = "vllm_hcu.v1.core.sched.scheduler.HcuScheduler"
UPSTREAM_SCHEDULER_PATH = f"{TARGET_MODULE}.Scheduler"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    scheduler = require_class(target, "Scheduler", TARGETS[0])
    if getattr(target, _MARKER, False):
        return False
    require_signature_prefix(
        require_callable(scheduler, "schedule", f"{TARGETS[0]}.schedule"),
        f"{TARGETS[0]}.schedule",
        ("self", "throttle_prefills"),
    )
    for method_name in (
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
        require_callable(
            scheduler,
            method_name,
            f"{TARGETS[0]}.{method_name}",
        )
    require_signature_prefix(
        require_callable(scheduler, "update_draft_token_ids", TARGETS[2]),
        TARGETS[2],
        ("self", "draft_token_ids"),
    )
    require_signature_prefix(
        require_callable(
            scheduler,
            "update_draft_token_ids_in_output",
            f"{TARGETS[0]}.update_draft_token_ids_in_output",
        ),
        f"{TARGETS[0]}.update_draft_token_ids_in_output",
        ("self", "draft_token_ids", "scheduler_output"),
    )
    original_update_draft_token_ids = require_callable(
        scheduler, "update_draft_token_ids", TARGETS[2]
    )
    original_make_stats = require_callable(scheduler, "make_stats", TARGETS[5])

    def hcu_dcut_truncate(self, spec_token_ids: list[int], keep_len: int):
        keep_len = max(0, int(keep_len))
        actual_keep_len = min(keep_len, len(spec_token_ids))
        self._pending_dcut_kept_draft_tokens = getattr(
            self, "_pending_dcut_kept_draft_tokens", 0
        ) + actual_keep_len + 1
        self._pending_dcut_total_draft_tokens = getattr(
            self, "_pending_dcut_total_draft_tokens", 0
        ) + len(spec_token_ids) + 1
        return spec_token_ids[:actual_keep_len]

    def hcu_make_or_update_dcut_stats(self, spec_decoding_stats):
        total = getattr(self, "_pending_dcut_total_draft_tokens", 0)
        if not getattr(self, "log_stats", False) or not total:
            return spec_decoding_stats
        if spec_decoding_stats is None:
            return None
        observe_dcut = getattr(spec_decoding_stats, "observe_dcut", None)
        if not callable(observe_dcut):
            raise PatchCompatibilityError(
                "SpecDecodingStats.observe_dcut was not installed"
            )
        observe_dcut(
            kept_draft_tokens=getattr(
                self, "_pending_dcut_kept_draft_tokens", 0
            ),
            total_draft_tokens=total,
        )
        self._pending_dcut_kept_draft_tokens = 0
        self._pending_dcut_total_draft_tokens = 0
        return spec_decoding_stats

    @functools.wraps(original_update_draft_token_ids)
    def hcu_update_draft_token_ids(self, draft_token_ids):
        result = original_update_draft_token_ids(self, draft_token_ids)
        keep_lens = getattr(draft_token_ids, "dcut_keep_lens", None)
        if keep_lens is None:
            return result
        if len(keep_lens) != len(draft_token_ids.req_ids):
            raise ValueError(
                "D-Cut keep-length count does not match request count: "
                f"{len(keep_lens)} != {len(draft_token_ids.req_ids)}"
            )
        for req_id, keep_len in zip(draft_token_ids.req_ids, keep_lens):
            request = self.requests.get(req_id)
            if (
                request is None
                or request.is_finished()
                or request.is_prefill_chunk
            ):
                continue
            request.spec_token_ids = self._dcut_truncate(
                request.spec_token_ids,
                keep_len,
            )
        return result

    @functools.wraps(original_make_stats)
    def hcu_make_stats(
        self,
        spec_decoding_stats=None,
        kv_connector_stats=None,
        cudagraph_stats=None,
        perf_stats=None,
    ):
        spec_decoding_stats = self._make_or_update_dcut_stats(
            spec_decoding_stats
        )
        return original_make_stats(
            self,
            spec_decoding_stats,
            kv_connector_stats,
            cudagraph_stats,
            perf_stats,
        )

    setattr(
        scheduler,
        "_vllm_hcu_original_dcut_update_draft_token_ids",
        original_update_draft_token_ids,
    )
    setattr(scheduler, "update_draft_token_ids", hcu_update_draft_token_ids)
    setattr(scheduler, "_dcut_truncate", hcu_dcut_truncate)
    setattr(
        scheduler,
        "_make_or_update_dcut_stats",
        hcu_make_or_update_dcut_stats,
    )
    setattr(scheduler, "_vllm_hcu_original_dcut_make_stats", original_make_stats)
    setattr(scheduler, "make_stats", hcu_make_stats)
    setattr(target, _MARKER, True)
    return True


def select_hcu_scheduler(vllm_config: object) -> bool:
    """Select HcuScheduler only for explicit split-P/D.

    Multi-layer MTP remains on the official scheduler and its existing
    ``DraftTokenIds`` channel.  This selector is safe to call in every process.
    """

    hcu_config = get_hcu_config(vllm_config)
    if hcu_config.enable_multi_layers_mtp:
        # The exact import callback performs the required-channel contract
        # check when vLLM resolves the scheduler module.
        pass
    if not henvs.VLLM_HCU_USE_PD_SPLIT:
        return False
    if not henvs.VLLM_HCU_USE_CUSTOM_OPS:
        raise RuntimeError(
            "VLLM_HCU_USE_PD_SPLIT requires VLLM_HCU_USE_CUSTOM_OPS; "
            "the required HCU scheduler was not enabled"
        )

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if scheduler_config is None or not hasattr(scheduler_config, "scheduler_cls"):
        raise PatchCompatibilityError(
            "vllm_config.scheduler_config.scheduler_cls is missing"
        )
    selected = scheduler_config.scheduler_cls
    if selected not in (None, UPSTREAM_SCHEDULER_PATH, HCU_SCHEDULER_PATH):
        raise RuntimeError(
            "split-P/D requires HcuScheduler but another scheduler_cls was selected: "
            f"{selected!r}"
        )

    if not hasattr(scheduler_config, "async_scheduling"):
        raise PatchCompatibilityError(
            "vllm_config.scheduler_config.async_scheduling is missing"
        )
    async_scheduling = scheduler_config.async_scheduling
    if async_scheduling is True:
        # vLLM v0.25.1 resolves its default async policy before invoking
        # Platform.check_and_update_config().  HcuScheduler intentionally
        # inherits the synchronous Scheduler and therefore does not implement
        # AsyncScheduler's placeholder/cache-update protocol.  Reject the
        # inconsistent pair instead of allowing silent request-state damage.
        raise RuntimeError(
            "split-P/D HcuScheduler does not support async scheduling in "
            "vLLM v0.25.1; explicitly disable it with --no-async-scheduling"
        )
    if async_scheduling is None:
        # This is reachable for direct/programmatic selector use before
        # VllmConfig.__post_init__.  Pin the only supported policy so the
        # subsequent v0.25.1 auto-selection cannot turn it back on.
        scheduler_config.async_scheduling = False
    elif async_scheduling is not False:
        raise PatchCompatibilityError(
            "vllm_config.scheduler_config.async_scheduling must be bool or None"
        )

    if selected == HCU_SCHEDULER_PATH:
        return False
    # A qualified string is the public lazy selection form supported by vLLM.
    # Resolving it here would import the full scheduler/model stack during
    # config construction.
    scheduler_config.scheduler_cls = HCU_SCHEDULER_PATH
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [
    "HCU_SCHEDULER_PATH",
    "PATCH_ID",
    "TARGET_MODULE",
    "TARGETS",
    "UPSTREAM_SCHEDULER_PATH",
    "apply",
    "apply_to_module",
    "select_hcu_scheduler",
]
