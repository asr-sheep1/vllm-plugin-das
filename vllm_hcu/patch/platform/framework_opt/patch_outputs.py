# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Retire ModelRunnerOutput IPC mutation in favour of DraftTokenIds."""

from __future__ import annotations

import functools
from copy import copy
from dataclasses import dataclass, fields, is_dataclass
from types import ModuleType

from ._common import PatchCompatibilityError, load_exact_module, require_class

TARGET_MODULE = "vllm.v1.outputs"
PATCH_ID = "platform.framework_opt.outputs_draft_token_ids"
TARGETS = (
    f"{TARGET_MODULE}.ModelRunnerOutput",
    f"{TARGET_MODULE}.DraftTokenIds",
    f"{TARGET_MODULE}.EMPTY_MODEL_RUNNER_OUTPUT",
)
_MARKER = "_vllm_hcu_draft_token_ids_contract_validated"


@dataclass
class _DcutDraftFields:
    """Serializable fields injected into v0.25.1 ``DraftTokenIds``."""

    dcut_keep_lens: list[int] | None = None


def _install_serializable_dcut_field(draft_ids: type) -> None:
    annotations = dict(getattr(draft_ids, "__annotations__", {}))
    dataclass_fields = getattr(draft_ids, "__dataclass_fields__", None)
    if not isinstance(dataclass_fields, dict):
        raise PatchCompatibilityError(
            "vLLM DraftTokenIds must remain a mutable dataclass"
        )
    source_field = fields(_DcutDraftFields)[0]
    annotations[source_field.name] = list[int] | None
    dataclass_fields[source_field.name] = copy(source_field)
    draft_ids.__annotations__ = annotations
    setattr(draft_ids, source_field.name, None)


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if getattr(target, _MARKER, False):
        return False
    model_output = require_class(target, "ModelRunnerOutput", TARGETS[0])
    draft_ids = require_class(target, "DraftTokenIds", TARGETS[1])
    if not is_dataclass(model_output) or not is_dataclass(draft_ids):
        raise PatchCompatibilityError("vLLM output contracts must be dataclasses")
    model_fields = {field.name for field in fields(model_output)}
    if "spec_token_ids" in model_fields:
        raise PatchCompatibilityError(
            "ModelRunnerOutput was source-patched with spec_token_ids; clean vLLM is required"
        )
    draft_fields = tuple(field.name for field in fields(draft_ids))
    if draft_fields != ("req_ids", "draft_token_ids"):
        raise PatchCompatibilityError(
            f"DraftTokenIds has incompatible fields: {draft_fields!r}"
        )
    original_draft_init = vars(draft_ids).get("__init__")
    if not callable(original_draft_init):
        raise PatchCompatibilityError("DraftTokenIds.__init__ is missing")

    @functools.wraps(original_draft_init)
    def hcu_dcut_draft_init(
        self,
        req_ids,
        draft_token_ids,
        dcut_keep_lens=None,
    ):
        original_draft_init(self, req_ids, draft_token_ids)
        self.dcut_keep_lens = dcut_keep_lens

    _install_serializable_dcut_field(draft_ids)
    setattr(draft_ids, "_vllm_hcu_original_dcut_init", original_draft_init)
    setattr(draft_ids, "__init__", hcu_dcut_draft_init)
    empty = getattr(target, "EMPTY_MODEL_RUNNER_OUTPUT", None)
    if not isinstance(empty, model_output):
        raise PatchCompatibilityError("EMPTY_MODEL_RUNNER_OUTPUT has wrong type")
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = [ "PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
