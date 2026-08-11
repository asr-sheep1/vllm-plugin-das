# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Backport D-Cut keep-ratio accounting to vLLM v0.25.1 metrics."""

from __future__ import annotations

import functools
from copy import copy
from dataclasses import dataclass, fields
from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_class

TARGET_MODULE = "vllm.v1.spec_decode.metrics"
PATCH_ID = "platform.framework_opt.spec_decode.dcut_metrics"
TARGETS = (
    f"{TARGET_MODULE}.SpecDecodingStats.observe_dcut",
    f"{TARGET_MODULE}.SpecDecodingLogging.observe",
    f"{TARGET_MODULE}.SpecDecodingLogging.log",
)
_MARKER = "_vllm_hcu_dcut_metrics_patch_applied"


@dataclass
class _DcutStatsFields:
    """Serializable fields injected into v0.25.1 SpecDecodingStats.

    EngineCoreOutputs crosses the process boundary through msgspec.  Dynamic
    instance attributes are silently omitted by msgspec's dataclass encoder,
    so merely attaching counters in ``observe_dcut`` is insufficient.  Add
    real dataclass fields while preserving the upstream class identity used by
    already-imported vLLM modules.
    """

    dcut_kept_draft_tokens: int = 0
    dcut_total_draft_tokens: int = 0


def _install_serializable_stats_fields(stats_class: type) -> None:
    annotations = dict(getattr(stats_class, "__annotations__", {}))
    dataclass_fields = getattr(stats_class, "__dataclass_fields__", None)
    if not isinstance(dataclass_fields, dict):
        raise PatchCompatibilityError(
            "vLLM SpecDecodingStats must remain a mutable dataclass"
        )
    for source_field in fields(_DcutStatsFields):
        name = source_field.name
        annotations[name] = int
        if name not in dataclass_fields:
            dataclass_fields[name] = copy(source_field)
        setattr(stats_class, name, 0)
    stats_class.__annotations__ = annotations


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    stats_class = require_class(target, "SpecDecodingStats", TARGETS[0])
    logging_class = require_class(target, "SpecDecodingLogging", TARGETS[1])
    _install_serializable_stats_fields(stats_class)

    original_reset = vars(logging_class).get("reset")
    original_observe = vars(logging_class).get("observe")
    original_log = vars(logging_class).get("log")
    if not all(callable(fn) for fn in (original_reset, original_observe, original_log)):
        raise PatchCompatibilityError(
            "vLLM speculative decoding logging contract is incompatible"
        )

    def observe_dcut(self, kept_draft_tokens: int, total_draft_tokens: int):
        if kept_draft_tokens < 0 or total_draft_tokens < kept_draft_tokens:
            raise ValueError(
                "invalid D-Cut accounting: "
                f"kept={kept_draft_tokens}, total={total_draft_tokens}"
            )
        self.dcut_kept_draft_tokens = getattr(
            self, "dcut_kept_draft_tokens", 0
        ) + int(kept_draft_tokens)
        self.dcut_total_draft_tokens = getattr(
            self, "dcut_total_draft_tokens", 0
        ) + int(total_draft_tokens)

    @functools.wraps(original_reset)
    def hcu_dcut_reset(self):
        original_reset(self)
        self.dcut_kept_draft_tokens = []
        self.dcut_total_draft_tokens = []

    @functools.wraps(original_observe)
    def hcu_dcut_observe(self, spec_decoding_stats):
        original_observe(self, spec_decoding_stats)
        self.dcut_kept_draft_tokens.append(
            getattr(spec_decoding_stats, "dcut_kept_draft_tokens", 0)
        )
        self.dcut_total_draft_tokens.append(
            getattr(spec_decoding_stats, "dcut_total_draft_tokens", 0)
        )

    @functools.wraps(original_log)
    def hcu_dcut_log(self, log_fn=None):
        if log_fn is None:
            log_fn = getattr(target, "logger").info
        kept = sum(self.dcut_kept_draft_tokens)
        total = sum(self.dcut_total_draft_tokens)
        original_log(self, log_fn=log_fn)
        if total:
            log_fn(
                "D-Cut metrics: Kept %d of %d target-forward slots "
                "(keep ratio: %.3f)",
                kept,
                total,
                kept / total,
            )

    setattr(stats_class, "observe_dcut", observe_dcut)
    setattr(logging_class, "_vllm_hcu_original_dcut_reset", original_reset)
    setattr(logging_class, "_vllm_hcu_original_dcut_observe", original_observe)
    setattr(logging_class, "_vllm_hcu_original_dcut_log", original_log)
    setattr(logging_class, "reset", hcu_dcut_reset)
    setattr(logging_class, "observe", hcu_dcut_observe)
    setattr(logging_class, "log", hcu_dcut_log)
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
