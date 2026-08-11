# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DSpark/DFly block proposer for the vLLM v0.25.1 HCU runner."""

import math
from dataclasses import replace

import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import (
    SpecDecodeBaseProposer,
    compute_probs_and_sample_next_token,
)
from vllm.v1.spec_decode.utils import next_power_of_2
from vllm.triton_utils import tl, triton

from .dfly_gate import is_dfly_speculative_config


logger = init_logger(__name__)

_DCUT_FALLBACK_RATIO = 3 / 4
_DCUT_RATIO_NUMS = (1, 2, 3, 4)
_DCUT_PROFILE_SEQ_LEN = 2048
_DCUT_PROFILE_WARMUPS = 3
_DCUT_PROFILE_STEPS = 10


@triton.jit
def copy_and_expand_dspark_inputs_kernel(
    next_token_ids_ptr,
    target_positions_ptr,
    out_input_ids_ptr,
    out_context_positions_ptr,
    out_query_positions_ptr,
    out_context_slot_mapping_ptr,
    out_query_slot_mapping_ptr,
    out_token_indices_ptr,
    block_table_ptr,
    block_table_stride,
    query_start_loc_ptr,
    num_rejected_tokens_ptr,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_tokens,
    total_input_tokens,
    BLOCK_SIZE: tl.constexpr,
    HAS_NUM_REJECTED: tl.constexpr = False,
    SAMPLE_FROM_ANCHOR: tl.constexpr = True,
):
    """Build context/query metadata for DFly's anchor-as-first layout."""
    req_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)
    ctx_start = tl.load(query_start_loc_ptr + req_idx)
    ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start
    total_tokens = num_ctx + num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = j < total_tokens
    is_ctx = j < num_ctx
    is_query = (~is_ctx) & in_bounds
    query_off = j - num_ctx

    ctx_pos_idx = tl.minimum(ctx_start + j, total_input_tokens - 1)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_ctx, other=0)
    if HAS_NUM_REJECTED:
        num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
        valid_ctx_end = ctx_end - num_rejected
    else:
        valid_ctx_end = ctx_end
    last_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_pos = last_pos + 1 + query_off
    positions = tl.where(is_ctx, ctx_pos, query_pos)

    ctx_pos_out = ctx_start + j
    tl.store(out_context_positions_ptr + ctx_pos_out, ctx_pos, mask=is_ctx)
    query_out = req_idx * num_query_per_req + query_off
    tl.store(out_query_positions_ptr + query_out, query_pos, mask=is_query)

    block_num = tl.minimum(positions // block_size, block_table_stride - 1)
    block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_num,
        mask=in_bounds,
        other=0,
    ).to(tl.int64)
    slot = block_id * block_size + (positions % block_size)
    tl.store(out_context_slot_mapping_ptr + ctx_pos_out, slot, mask=is_ctx)
    tl.store(out_query_slot_mapping_ptr + query_out, slot, mask=is_query)

    bonus_token = tl.load(next_token_ids_ptr + req_idx)
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)
    tl.store(out_input_ids_ptr + query_out, input_id, mask=is_query)

    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    is_sample = is_query & (query_off >= sample_off)
    sample_out_idx = req_idx * num_speculative_tokens + (query_off - sample_off)
    tl.store(
        out_token_indices_ptr + sample_out_idx,
        query_out,
        mask=is_sample,
    )


@triton.jit
def _token_logprob_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute one FP32 token log-probability per logits row."""

    row = tl.program_id(0)
    row_ptr = logits_ptr + row * logits_stride

    max_value = float("-inf")
    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            row_ptr + offsets,
            mask=offsets < vocab_size,
            other=float("-inf"),
        )
        max_value = tl.max(tl.maximum(values, max_value))
    max_value = max_value.to(tl.float32)

    exp_sum = 0.0
    for start in range(0, vocab_size, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(
            row_ptr + offsets,
            mask=offsets < vocab_size,
            other=0.0,
        ).to(tl.float32)
        exponentials = tl.exp(values - max_value)
        exp_sum += tl.sum(
            tl.where(offsets < vocab_size, exponentials, 0.0)
        )

    token_id = tl.load(token_ids_ptr + row)
    selected = tl.load(row_ptr + token_id).to(tl.float32)
    tl.store(output_ptr + row, selected - max_value - tl.log(exp_sum))


def _token_logprobs_from_logits(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Return accurate log-probabilities without an FP32 logits temporary."""

    num_rows, vocab_size = logits.shape
    if not logits.is_cuda:
        logits_fp32 = logits.float()
        shifted = logits_fp32 - logits_fp32.max(dim=-1, keepdim=True).values
        denominator = shifted.exp().sum(dim=-1, dtype=torch.float32).log()
        selected = shifted.gather(1, token_ids.unsqueeze(1)).squeeze(1).float()
        return selected - denominator

    if logits.stride(1) != 1:
        raise ValueError("DFly logits vocabulary dimension must be contiguous")
    output = logits.new_empty((num_rows,), dtype=torch.float32)
    _token_logprob_kernel[(num_rows,)](
        output,
        logits,
        logits.stride(0),
        token_ids.to(torch.int64),
        vocab_size,
        BLOCK_SIZE=min(8192, next_power_of_2(vocab_size)),
    )
    return output


class DSparkProposer(DFlashProposer):
    """Parallel block forward followed by left-to-right DFly sampling."""

    def _init_parallel_drafting_params(self) -> None:
        """Resolve DFlash/DFly mask-token aliases supported by newer vLLM."""
        hf_config = self.draft_model_config.hf_config
        dflash_config = getattr(hf_config, "dflash_config", None) or {}
        dflare_config = getattr(hf_config, "dflare_config", None) or {}
        candidates = (
            dflash_config.get("mask_token_id"),
            dflare_config.get("mask_token_id"),
            getattr(hf_config, "mask_token_id", None),
            getattr(hf_config, "dspark_noise_token_id", None),
            getattr(hf_config, "pard_token", None),
            getattr(hf_config, "ptd_token_id", None),
        )
        token_id = next((value for value in candidates if value is not None), None)
        if token_id is None:
            raise ValueError(
                "DFly parallel drafting requires a mask token in "
                "dflash_config, dflare_config, mask_token_id, "
                "dspark_noise_token_id, pard_token, or ptd_token_id"
            )
        self.parallel_drafting_token_id = int(token_id)
        if self.pass_hidden_states_to_model:
            self.parallel_drafting_hidden_state_tensor = torch.empty(
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )

    @property
    def dflash_config(self) -> dict:
        hf_config = self.draft_model_config.hf_config
        return (
            getattr(hf_config, "dflash_config", None)
            or getattr(hf_config, "dflare_config", None)
            or {}
        )

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        if not is_dfly_speculative_config(vllm_config.speculative_config):
            raise NotImplementedError(
                "The HCU V1 DSpark-style proposer currently supports only "
                "Qwen3DFlyModel. Use vLLM's V2 runner for native DSpark."
            )

        # DFlashProposer.__init__ in v0.25.1 rejects method="dspark".  Run
        # its base initializer and reproduce the DFlash-owned buffers here.
        SpecDecodeBaseProposer.__init__(
            self,
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        self._runner = runner
        # v0.25.1 subtracts one hidden-state slot for every method other than
        # DFlash. DSpark uses DFlash's query-block ABI too, so no slot is
        # recycled here; retain all N parallel draft slots.
        self.net_num_new_slots_per_request = self.extra_slots_per_request
        self.needs_extra_input_slots = True
        hf_config = self.draft_model_config.hf_config
        self.sample_from_anchor = not bool(
            getattr(hf_config, "dspark_bonus_anchor", False)
        )
        self.num_query_per_req = (
            self.num_speculative_tokens
            if self.sample_from_anchor
            else 1 + self.num_speculative_tokens
        )
        self.max_query_tokens = self.max_batch_size * (
            1 + self.num_speculative_tokens
        )
        self.max_positions = self.max_num_tokens + self.max_query_tokens
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens, dtype=torch.int64, device=device
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self.positions = torch.zeros(
            self.max_query_tokens, dtype=torch.int64, device=device
        )
        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )
        self.parallel_drafting_hidden_state_tensor = None
        self.dflash_causal = False

        block_size = getattr(hf_config, "block_size", None)
        self.markov_position_offset = int(
            not self.sample_from_anchor
            and bool(getattr(hf_config, "markov_pos_adaptive", False))
            and block_size == self.num_query_per_req
        )
        self._anchor_idx = (
            torch.arange(self.max_batch_size, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        # DFly consumes a concatenation of several target-layer hidden states
        # when projecting the context K/V.  Keep the dummy tensor alive for the
        # lifetime of the proposer: allocating it in dummy_run records a
        # short-lived address in CUDA graphs and causes a VM fault as soon as
        # capture synchronizes or the graph is replayed.
        num_target_layers = getattr(hf_config, "num_target_layers", None)
        if num_target_layers is None:
            num_target_layers = len(getattr(hf_config, "target_layer_ids", ()))
        if not num_target_layers:
            raise ValueError(
                "DFly requires num_target_layers or target_layer_ids for "
                "its persistent CUDA-graph context buffer"
            )
        self._dummy_context_hidden_states = torch.zeros(
            (
                self.max_num_tokens,
                int(num_target_layers) * self.hidden_size,
            ),
            dtype=self.dtype,
            device=device,
        )
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None
        self._dcut_keep_lens_cache: torch.Tensor | None = None
        self._dcut_costs_by_bs: dict[int, torch.Tensor] = {}
        self._dcut_keep_counts: torch.Tensor | None = None

    def _draft_model(self):
        model = self.model
        if isinstance(model, BreakableCUDAGraphWrapper):
            model = model.unwrap()
        return model

    def load_model(self, target_model) -> None:
        super().load_model(target_model)
        model = self._draft_model()
        d2t = getattr(model, "draft_id_to_target_id", None)
        if self._enable_probabilistic_draft_probs and d2t is not None:
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            self._draft_scatter_buf = torch.full(
                (self.max_batch_size, self.target_vocab_size),
                float("-inf"),
                dtype=self.dtype,
                device=self.device,
            )

    def model_returns_tuple(self) -> bool:
        return False

    def propose(self, *args, **kwargs) -> torch.Tensor:
        if "target_hidden_states" in kwargs:
            target_hidden_states = kwargs["target_hidden_states"]
            kwargs["target_hidden_states"] = self._draft_model().combine_hidden_states(
                target_hidden_states
            )
        elif len(args) >= 4:
            mutable = list(args)
            mutable[3] = self._draft_model().combine_hidden_states(mutable[3])
            args = tuple(mutable)
        return super().propose(*args, **kwargs)

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_total = batch_size * self.num_query_per_req
        self._dflash_num_context = num_context
        self._dflash_hidden_states = target_hidden_states
        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        max_tokens_per_req = cad.max_query_len + self.num_query_per_req
        kernel_block_size = min(256, next_power_of_2(max_tokens_per_req))
        num_blocks = (
            max_tokens_per_req + kernel_block_size - 1
        ) // kernel_block_size
        has_num_rejected = num_rejected_tokens_gpu is not None
        copy_and_expand_dspark_inputs_kernel[(batch_size, num_blocks)](
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=self.num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            BLOCK_SIZE=kernel_block_size,
            HAS_NUM_REJECTED=has_num_rejected,
            SAMPLE_FROM_ANCHOR=self.sample_from_anchor,
        )

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu
        upper_bound = (
            cad.seq_lens_cpu_upper_bound + self.num_query_per_req
            if cad.seq_lens_cpu_upper_bound is not None
            else None
        )
        new_cad = CommonAttentionMetadata(
            query_start_loc=(
                self.arange[: batch_size + 1] * self.num_query_per_req
            ),
            seq_lens=effective_seq_lens + self.num_query_per_req,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * self.num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            seq_lens_cpu_upper_bound=upper_bound,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=self.num_query_per_req,
            max_seq_len=cad.max_seq_len + self.num_query_per_req,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=self._slot_mapping_buffer[:num_query_total],
            causal=False,
        )
        return num_query_total, token_indices_to_sample, new_cad

    def _scatter_to_target(
        self, draft_logits: torch.Tensor, num_reqs: int
    ) -> torch.Tensor:
        if self._d2t_scatter_index is None:
            return draft_logits
        assert self._draft_scatter_buf is not None
        output = self._draft_scatter_buf[:num_reqs]
        output.fill_(float("-inf"))
        output.index_copy_(1, self._d2t_scatter_index, draft_logits.to(output.dtype))
        return output

    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._dcut_keep_lens_cache = None
        model = self._draft_model()
        n_spec = self.num_speculative_tokens
        num_reqs = hidden_states.shape[0] // n_spec
        hidden_per_step = hidden_states.view(num_reqs, n_spec, -1)
        use_correction = bool(
            getattr(model, "has_hidden_correction", lambda: False)()
        )
        use_markov = bool(getattr(model, "has_markov", lambda: True)())
        base_logits = None
        if not use_correction:
            base_logits = model.compute_draft_logits(hidden_states).view(
                num_reqs, n_spec, -1
            )

        prev = self.input_ids[self._anchor_idx[:num_reqs]].long()
        draft_tokens = torch.empty(
            (num_reqs, n_spec), dtype=torch.int64, device=self.device
        )
        probabilistic = (
            self._enable_probabilistic_draft_probs
            and not sampling_metadata.all_greedy
        )
        probs: list[torch.Tensor] | None = [] if probabilistic else None
        speculative_config = getattr(self, "speculative_config", None)
        collect_dcut_logprobs = (
            getattr(speculative_config, "dflash_dcut_mode", "off") != "off"
        )
        dcut_logprobs: list[torch.Tensor] | None = (
            [] if collect_dcut_logprobs else None
        )

        for step in range(n_spec):
            if use_correction:
                corrected = model.apply_hidden_correction(
                    hidden_per_step[:, step], prev
                )
                logits = model.compute_draft_logits(corrected)
            else:
                assert base_logits is not None
                logits = base_logits[:, step]
            if use_markov:
                logits = logits + model.markov_bias(
                    model.markov_embed(prev),
                    step=step + self.markov_position_offset,
                )

            if probabilistic:
                target_logits = self._scatter_to_target(logits, num_reqs)
                sampled, step_probs = compute_probs_and_sample_next_token(
                    target_logits,
                    sampling_metadata,
                    self.use_fp64_gumbel,
                )
                assert probs is not None
                probs.append(step_probs)
                if dcut_logprobs is not None:
                    selected_probs = step_probs.gather(
                        1, sampled.unsqueeze(1)
                    ).squeeze(1)
                    dcut_logprobs.append(
                        torch.log(selected_probs.float().clamp_min_(1e-30))
                    )
            else:
                draft_sampled = logits.argmax(dim=-1)
                sampled = model.map_draft_to_target(draft_sampled)
                if dcut_logprobs is not None:
                    dcut_logprobs.append(
                        _token_logprobs_from_logits(logits, draft_sampled)
                    )
            draft_tokens[:, step] = sampled
            prev = sampled

        flat_probs = None
        if probs is not None:
            flat_probs = torch.stack(probs, dim=1).reshape(
                num_reqs * n_spec, -1
            )
        if dcut_logprobs is not None:
            self._dcut_keep_lens_cache = self._select_dcut_keep_lens(
                torch.stack(dcut_logprobs, dim=1)
            )
        return draft_tokens.reshape(-1), flat_probs

    @staticmethod
    def _get_dcut_keep_count(
        batch_size: int,
        num_draft_tokens: int,
        ratio: float,
    ) -> int:
        """Return the global draft budget after reserving one bonus per req."""

        return max(
            0,
            math.ceil(batch_size * (num_draft_tokens + 1) * ratio) - batch_size,
        )

    def _select_dcut_keep_lens(self, logprobs: torch.Tensor) -> torch.Tensor:
        """Select prefix lengths using a cross-request global confidence budget."""

        batch_size, num_draft_tokens = logprobs.shape
        cumulative_logprobs = logprobs.cumsum(dim=1).flatten()
        mode = getattr(self.speculative_config, "dflash_dcut_mode", "off")
        if mode == "off":
            return torch.full(
                (batch_size,),
                num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )

        num_keep: int | torch.Tensor
        if mode == "fixed_ratio":
            ratio = float(self.speculative_config.dflash_dcut)
            num_keep = self._get_dcut_keep_count(
                batch_size, num_draft_tokens, ratio
            )
            if num_keep == 0:
                return torch.zeros(
                    (batch_size,), dtype=torch.int32, device=self.device
                )
            _, selected_indices = torch.topk(cumulative_logprobs, k=num_keep)
        else:
            profile_bs = min(
                (size for size in self._dcut_costs_by_bs if size >= batch_size),
                default=None,
            )
            if profile_bs is None or self._dcut_keep_counts is None:
                logger.warning_once(
                    "DFly D-Cut selector has no profiled cost for bs=%d; "
                    "using fallback ratio %.2f.",
                    batch_size,
                    _DCUT_FALLBACK_RATIO,
                )
                num_keep = self._get_dcut_keep_count(
                    batch_size,
                    num_draft_tokens,
                    _DCUT_FALLBACK_RATIO,
                )
                if num_keep == 0:
                    return torch.zeros(
                        (batch_size,), dtype=torch.int32, device=self.device
                    )
                _, selected_indices = torch.topk(
                    cumulative_logprobs, k=num_keep
                )
            else:
                keep_counts = self._dcut_keep_counts[batch_size]
                costs = self._dcut_costs_by_bs[profile_bs]
                sorted_logprobs, sorted_indices = torch.sort(
                    cumulative_logprobs, descending=True
                )
                prefix_scores = torch.cumsum(torch.exp(sorted_logprobs), dim=0)
                candidate_scores = torch.zeros_like(costs)
                valid = keep_counts > 0
                candidate_scores[valid] = prefix_scores[keep_counts[valid] - 1]
                # Every request emits one bonus token regardless of its kept
                # draft length, so optimize total output tokens per unit cost.
                candidate_scores += batch_size
                best = torch.argmax(candidate_scores / costs)
                num_keep = keep_counts[best]
                selection_mask = (
                    torch.arange(
                        batch_size * num_draft_tokens,
                        device=self.device,
                    )
                    < num_keep
                )
                selected_indices = sorted_indices[selection_mask]

        keep_lens = torch.zeros(
            (batch_size,), dtype=torch.int32, device=self.device
        )
        updates = torch.ones_like(selected_indices, dtype=torch.int32)
        keep_lens.scatter_add_(
            0,
            selected_indices // num_draft_tokens,
            updates,
        )
        return keep_lens

    def take_dcut_keep_lens(self) -> torch.Tensor | None:
        return self._dcut_keep_lens_cache

    def profile_dcut_cost_table(self) -> None:
        """Profile target verification plus full DFly drafting cost by ratio."""

        if self._runner is None:
            raise RuntimeError("DFly D-Cut auto profiling requires the HCU runner")
        costs_by_bs: dict[int, list[tuple[int, float]]] = {}
        max_bs = min(self.max_batch_size, self._runner.max_num_reqs)
        batch_sizes = torch.arange(
            max_bs + 1, device=self.device, dtype=torch.long
        )[:, None]
        ratio_nums = torch.tensor(
            _DCUT_RATIO_NUMS, device=self.device, dtype=torch.long
        )
        keep_counts = torch.div(
            batch_sizes
            * (self.num_speculative_tokens + 1)
            * ratio_nums
            + 3,
            4,
            rounding_mode="floor",
        )
        self._dcut_keep_counts = torch.clamp(keep_counts - batch_sizes, min=0)

        for batch_size in self._get_dcut_profile_batch_sizes():
            entries: list[tuple[int, float]] = []
            full_draft_tokens = batch_size * self.num_query_per_req
            assert self._dcut_keep_counts is not None
            # Make the selector take its production auto path while the four
            # candidates for this batch size are being profiled.
            self._dcut_costs_by_bs[batch_size] = torch.ones(
                len(_DCUT_RATIO_NUMS),
                dtype=torch.float32,
                device=self.device,
            )
            for keep_count in self._dcut_keep_counts[batch_size].tolist():
                target_tokens = batch_size + keep_count
                cost = self._profile_dcut_full_cost_ms(
                    batch_size=batch_size,
                    target_tokens=target_tokens,
                    draft_tokens=full_draft_tokens,
                )
                entries.append((keep_count, cost))
            profiled = torch.tensor(
                [entry[1] for entry in entries],
                dtype=torch.float32,
                device=self.device,
            )
            costs = profiled.cummax(dim=0).values
            self._dcut_costs_by_bs[batch_size] = costs
            costs_by_bs[batch_size] = list(
                zip([entry[0] for entry in entries], costs.tolist())
            )
        if costs_by_bs:
            logger.info("DFly D-Cut warmup full-cost table: %s", costs_by_bs)

    def _get_dcut_profile_batch_sizes(self) -> tuple[int, ...]:
        verify_tokens_per_request = 1 + self.num_speculative_tokens
        assert self._runner is not None
        max_bs = min(
            self.max_batch_size,
            self._runner.max_num_reqs,
            self._runner.max_num_tokens // verify_tokens_per_request,
        )
        if max_bs <= 0:
            return ()
        capture_sizes = self.compilation_config.cudagraph_capture_sizes or []
        sizes = [
            capture_size // verify_tokens_per_request
            for capture_size in capture_sizes
            if capture_size % verify_tokens_per_request == 0
            and 0 < capture_size // verify_tokens_per_request <= max_bs
        ]
        sizes.append(max_bs)
        return tuple(sorted(set(sizes)))

    def _profile_dcut_full_cost_ms(
        self,
        *,
        batch_size: int,
        target_tokens: int,
        draft_tokens: int,
    ) -> float:
        assert self._runner is not None
        profile_seq_len = min(_DCUT_PROFILE_SEQ_LEN, self._runner.max_model_len)
        dummy_run_kwargs = dict(
            force_attention=True,
            allow_microbatching=False,
            skip_eplb=True,
            is_profile=False,
            dcut_profile_num_reqs=batch_size,
            drafter_dummy_num_tokens=draft_tokens,
            profile_seq_lens=profile_seq_len,
        )
        for _ in range(_DCUT_PROFILE_WARMUPS):
            self._runner._dummy_run(target_tokens, **dummy_run_kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(_DCUT_PROFILE_STEPS):
            self._runner._dummy_run(target_tokens, **dummy_run_kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / _DCUT_PROFILE_STEPS

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
        num_query_tokens: int | None = None,
        profile_num_reqs: int | None = None,
    ) -> None:
        requested_query_tokens = (
            num_tokens if num_query_tokens is None else num_query_tokens
        )
        num_query_tokens = min(requested_query_tokens, self.max_query_tokens)
        runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        model = self._draft_model()
        context_states = self._dummy_context_hidden_states[:num_tokens]
        model.precompute_and_store_context_kv(
            context_states,
            self._context_positions_buffer[:num_tokens],
        )
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            hidden_states = self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )
        if profile_num_reqs is not None:
            self._dummy_sample_run(hidden_states, profile_num_reqs)

    def _dummy_sample_run(
        self,
        hidden_states: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Include DFly correction, LM head, and selection in cost profiles."""

        num_sample_tokens = num_reqs * self.num_speculative_tokens
        if hidden_states.shape[0] < num_sample_tokens:
            return
        sample_hidden_states = torch.rand_like(hidden_states[:num_sample_tokens])
        self._sample_draft_tokens(
            sample_hidden_states,
            self._make_dummy_sampling_metadata(num_reqs),
        )

    def _make_dummy_sampling_metadata(
        self,
        num_reqs: int,
    ) -> SamplingMetadata:
        dummy_tensors = lambda value: torch.full(  # noqa: E731
            (num_reqs,), value, device=self.device
        )
        return SamplingMetadata(
            temperature=torch.full((1,), 0.5, device=self.device),
            all_greedy=not self._enable_probabilistic_draft_probs,
            all_random=False,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            logprob_token_ids=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.0),
            presence_penalties=dummy_tensors(0.0),
            repetition_penalties=dummy_tensors(1.0),
            output_token_ids=[[] for _ in range(num_reqs)],
            spec_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )

    def _create_draft_vllm_config(self) -> VllmConfig:
        base = SpecDecodeBaseProposer._create_draft_vllm_config(self)
        arch = base.model_config.model_arch_config
        if arch.is_mm_prefix_lm:
            base.model_config.model_arch_config = replace(
                arch, is_mm_prefix_lm=False
            )
        return replace(
            base,
            attention_config=replace(base.attention_config, use_non_causal=True),
        )


__all__ = ["DSparkProposer", "copy_and_expand_dspark_inputs_kernel"]
