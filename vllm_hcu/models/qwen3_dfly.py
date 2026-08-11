# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU integration for the Qwen3 DFly speculative draft model.

DFly reuses vLLM's DFlash context-KV backbone and the DSpark sequential block
runtime.  Unlike DSpark's Markov head, it fuses several target-model hidden
states per draft layer and applies a previous-token-conditioned SwiGLU
correction before sampling.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    maybe_prefix,
    process_eagle_weight,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _fused_dfly_context_rmsnorm_kernel(
    base_ptr,
    stacked_ptr,
    fusion_ptr,
    norm_weight_ptr,
    output_ptr,
    num_ctx,
    hidden_size: tl.constexpr,
    num_target_layers: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse target-layer mixing, residual add, and shared RMSNorm."""
    row = tl.program_id(0)
    layer = row // num_ctx
    ctx = row - layer * num_ctx
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    values = tl.load(
        base_ptr + ctx * hidden_size + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    stacked_row = ctx * num_target_layers * hidden_size
    fusion_row = layer * num_target_layers
    for target_idx in range(num_target_layers):
        target = tl.load(
            stacked_ptr + stacked_row + target_idx * hidden_size + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        coefficient = tl.load(fusion_ptr + fusion_row + target_idx).to(tl.float32)
        values += coefficient * target

    variance = tl.sum(values * values, axis=0) / hidden_size
    values *= tl.rsqrt(variance + eps)
    norm_weight = tl.load(
        norm_weight_ptr + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(output_ptr + row * hidden_size + offsets, values * norm_weight, mask=mask)


class HiddenStatesCorrection(nn.Module):
    """Previous-token-conditioned residual SwiGLU correction used by DFly."""

    def __init__(
        self,
        hidden_size: int,
        embed_size: int,
        intermediate_size: int,
        rms_norm_eps: float = 1e-6,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.embed_norm = RMSNorm(embed_size, eps=rms_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size + embed_size,
            [intermediate_size] * 2,
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_norm(hidden_states)
        embeds = self.embed_norm(prev_token_embeds.to(hidden_states.dtype))
        gate_up, _ = self.gate_up_proj(torch.cat([hidden, embeds], dim=-1))
        delta, _ = self.down_proj(self.act_fn(gate_up))
        return hidden_states + delta.to(hidden_states.dtype)


class Qwen3DFlyModel(DFlashQwen3Model):
    """DFlash backbone with per-draft-layer target-state fusion."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        config = self.config
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        if target_hidden_size != config.hidden_size:
            raise ValueError(
                "DFly requires target_hidden_size == hidden_size, got "
                f"{target_hidden_size} and {config.hidden_size}"
            )

        drafter_config: dict = {}
        drafter_config.update(getattr(config, "dflash_config", None) or {})
        drafter_config.update(getattr(config, "dflare_config", None) or {})
        target_layer_ids = (
            drafter_config.get("target_layer_ids")
            or getattr(config, "target_layer_ids", None)
            or getattr(config, "eagle_aux_hidden_state_layer_ids", None)
        )
        configured_num_target_layers = getattr(config, "num_target_layers", None)
        if target_layer_ids:
            num_target_layers = len(target_layer_ids)
            if (
                configured_num_target_layers is not None
                and int(configured_num_target_layers) != num_target_layers
            ):
                raise ValueError(
                    "DFly num_target_layers does not match target_layer_ids: "
                    f"{configured_num_target_layers} != {num_target_layers}"
                )
        elif configured_num_target_layers is not None:
            num_target_layers = int(configured_num_target_layers)
        else:
            raise ValueError("DFly requires num_target_layers or target_layer_ids")

        expected_fc_input = num_target_layers * target_hidden_size
        if self.fc.input_size != expected_fc_input:
            raise ValueError(
                "DFly context projection width mismatch: "
                f"fc.input_size={self.fc.input_size}, expected={expected_fc_input}"
            )

        self.num_target_layers = num_target_layers
        self.num_draft_layers = config.num_hidden_layers
        self.layer_fusion_weights = nn.Parameter(
            torch.empty(
                self.num_draft_layers,
                self.num_target_layers,
                dtype=vllm_config.model_config.dtype,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "_fusion_probs",
            torch.empty_like(self.layer_fusion_weights),
            persistent=False,
        )

        correction_type = str(
            getattr(config, "hidden_correction_type", "swiglu")
        ).lower()
        if correction_type != "swiglu":
            raise ValueError(
                "Qwen3DFlyModel only supports SwiGLU hidden correction, got "
                f"{correction_type!r}"
            )
        intermediate = int(
            getattr(config, "hidden_correction_intermediate_size", None)
            or config.hidden_size
        )
        self.hidden_correction = HiddenStatesCorrection(
            hidden_size=int(config.hidden_size),
            embed_size=int(config.hidden_size),
            intermediate_size=intermediate,
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
            prefix=maybe_prefix(prefix, "hidden_correction"),
        )

    def _build_fused_kv_buffers(self) -> None:
        super()._build_fused_kv_buffers()
        self._fusion_probs.copy_(
            F.softmax(self.layer_fusion_weights.float(), dim=-1).to(
                dtype=self.layer_fusion_weights.dtype
            )
        )

    def _build_context_kv_buffers(
        self,
        layers_attn: list[nn.Module],
        has_bias: bool,
    ) -> None:
        """Build logical ``[K, V]`` weights from either HCU linear layout.

        vLLM's DFlash implementation assumes an ``[out, in]`` linear weight
        and slices ``weight[q_size:]``.  HCU's default ``VLLM_USE_NN`` path
        stores unquantized linear weights as ``[in, out]`` so GEMMs can use
        them without a runtime transpose.  Slicing that physical layout on
        dimension 0 retains part of the input dimension and produces a fused
        tensor that cannot be reshaped as layer-major K/V weights.

        Normalize only this small derived buffer back to vLLM's logical
        ``[out, in]`` representation.  The live QKV parameters keep their HCU
        layout and the regular draft-layer forward path remains unchanged.
        """
        self._hidden_norm_weight = self.hidden_norm.weight.data

        kv_weights: list[torch.Tensor] = []
        for attn in layers_attn:
            weight = attn.qkv_proj.weight.data
            logical_output_size = attn.q_size + 2 * attn.kv_size
            standard_shape = (logical_output_size, self.config.hidden_size)
            nn_shape = (self.config.hidden_size, logical_output_size)
            if tuple(weight.shape) == standard_shape:
                kv_weight = weight[attn.q_size :]
            elif tuple(weight.shape) == nn_shape:
                kv_weight = weight[:, attn.q_size :].transpose(0, 1)
            else:
                raise ValueError(
                    "Unsupported DFly QKV weight layout: "
                    f"got {tuple(weight.shape)}, expected {standard_shape} "
                    f"or {nn_shape}"
                )
            kv_weights.append(kv_weight.contiguous())
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)

        if has_bias:
            kv_biases = [
                attn.qkv_proj.bias.data[attn.q_size :] for attn in layers_attn
            ]
            self._fused_kv_bias = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        self._k_norm_weights = torch.stack(
            [attn.k_norm.weight.data for attn in layers_attn], dim=0
        ).contiguous()

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        expected_width = self.num_target_layers * self.config.hidden_size
        actual_width = aux_hidden_states.shape[-1]
        if actual_width < expected_width:
            raise ValueError(
                "DFly received too few target hidden states: "
                f"width={actual_width}, expected at least {expected_width}"
            )
        return aux_hidden_states[..., :expected_width]

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        """Apply each draft layer's K norm with its own 1-D weight.

        v0.25.1 passes the stacked ``[L, head_dim]`` weights to an RMSNorm
        kernel whose ABI accepts one weight vector.  That silently gives the
        wrong normalization for every layer except the one selected by the
        backend.  DFly always has per-layer K-norm parameters, so issue one
        normalization per draft layer.
        """
        all_k_normed = torch.empty_like(all_k)
        for layer_idx, weight in enumerate(self._k_norm_weights):
            ops.rms_norm(
                all_k_normed[layer_idx],
                all_k[layer_idx],
                weight,
                self._rms_norm_eps,
            )
        return all_k_normed

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_size = self.config.hidden_size
        if num_layers != self.num_draft_layers:
            raise ValueError(
                f"DFly expected {self.num_draft_layers} draft layers, got {num_layers}"
            )
        expected_width = self.num_target_layers * hidden_size
        if context_states.shape[-1] != expected_width:
            raise ValueError(
                "DFly context width mismatch: "
                f"got {context_states.shape[-1]}, expected {expected_width}"
            )

        stacked = context_states.view(num_ctx, self.num_target_layers, hidden_size)
        base_context = self.fc(context_states)
        fusion_probs = self._fusion_probs.to(dtype=context_states.dtype)
        if context_states.is_cuda:
            normed = torch.empty(
                (self.num_draft_layers, num_ctx, hidden_size),
                dtype=context_states.dtype,
                device=context_states.device,
            )
            _fused_dfly_context_rmsnorm_kernel[
                (self.num_draft_layers * num_ctx,)
            ](
                base_context,
                stacked,
                fusion_probs,
                self._hidden_norm_weight,
                normed,
                num_ctx=num_ctx,
                hidden_size=hidden_size,
                num_target_layers=self.num_target_layers,
                eps=self._rms_norm_eps,
                BLOCK_SIZE=triton.next_power_of_2(hidden_size),
            )
        else:
            residual_context = torch.einsum("lt,ntd->lnd", fusion_probs, stacked)
            layer_context = residual_context + base_context.unsqueeze(0)
            layer_context_flat = layer_context.reshape(
                self.num_draft_layers * num_ctx, hidden_size
            )
            normed_flat = torch.empty_like(layer_context_flat)
            ops.rms_norm(
                normed_flat,
                layer_context_flat,
                self._hidden_norm_weight,
                self._rms_norm_eps,
            )
            normed = normed_flat.view(
                self.num_draft_layers, num_ctx, hidden_size
            )

        kv_size_per_partition = 2 * num_kv_heads * head_dim
        stacked_weight = self._fused_kv_weight.view(
            self.num_draft_layers, kv_size_per_partition, hidden_size
        )
        all_kv_flat = torch.bmm(normed, stacked_weight.transpose(1, 2))
        if self._fused_kv_bias is not None:
            all_kv_flat = all_kv_flat + self._fused_kv_bias.view(
                self.num_draft_layers, 1, kv_size_per_partition
            )
        all_kv = (
            all_kv_flat.view(
                self.num_draft_layers,
                num_ctx,
                2,
                num_kv_heads,
                head_dim,
            )
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        return all_kv[0], all_kv[1]


class Qwen3DFlyForCausalLM(Qwen3DSparkForCausalLM):
    """Top-level DFly draft wrapper registered by the HCU plugin."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3DFlyModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def has_markov(self) -> bool:
        return False

    def has_hidden_correction(self) -> bool:
        return True

    def apply_hidden_correction(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        prev_embeds = self.model.embed_input_ids(prev_token_ids)
        return self.model.hidden_correction(hidden_states, prev_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights: dict[str, torch.Tensor] = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        includes_lm_head = False

        for name, loaded_weight in weights:
            if (
                "t2d" in name
                or "confidence_head" in name
                or "markov_head" in name
                or "markov_w" in name
            ):
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name

            if name == "model.context_proj.weight":
                name = "model.fc.weight"
            elif name == "model.context_norm.weight":
                name = "model.hidden_norm.weight"
            elif name == "model.final_norm.weight":
                name = "model.norm.weight"

            includes_embed_tokens |= "embed_tokens" in name
            includes_lm_head |= "lm_head" in name
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_param_names: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        remaining_weights: list[tuple[str, torch.Tensor]] = []

        for name, loaded_weight in model_weights.items():
            if "scale" in name:
                remapped = maybe_remap_kv_scale_name(name, params_dict)
                if remapped is None:
                    continue
                name = remapped

            if "attention_sink_bias" in name:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                heads_per_rank = loaded_weight.shape[0] // tp_size
                param.data.copy_(
                    loaded_weight.narrow(0, tp_rank * heads_per_rank, heads_per_rank)
                )
                loaded_param_names.add(name)
                continue

            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                params_dict[mapped].weight_loader(
                    params_dict[mapped], loaded_weight, shard_id
                )
                loaded_param_names.add(mapped)
                matched = True
                break
            if matched:
                continue

            if name in params_dict:
                param = params_dict[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
                loaded_param_names.add(name)
            else:
                remaining_weights.append((name, loaded_weight))

        skip_substrs = ["mask_embedding", "confidence_head"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if remaining_weights:
            loaded_param_names.update(
                AutoWeightsLoader(self, skip_substrs=skip_substrs).load_weights(
                    iter(remaining_weights)
                )
            )

        missing = [
            name
            for name, param in params_dict.items()
            if name not in loaded_param_names
            and not any(skip in name for skip in skip_substrs)
            and param.numel() > 0
            and "_fusion_probs" not in name
        ]
        if missing:
            raise ValueError(
                "DFly checkpoint is missing required parameters after optional "
                "shared weights were excluded: "
                f"{missing[:12]}"
            )
        self.model._build_fused_kv_buffers()


__all__ = ["Qwen3DFlyModel", "Qwen3DFlyForCausalLM"]
