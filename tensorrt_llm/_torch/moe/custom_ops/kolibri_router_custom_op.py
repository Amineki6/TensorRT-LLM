# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused MoE Routing Kernel for Kolibri 1 (78B MoE).

Kolibri 1 routing performs:
    1. scores = x + e_score_correction_bias (logit-space bias)
    2. topk_ids = TopK(scores, k=6) over 384 experts
    3. topk_weights = sigmoid(x[topk_ids]) (unnormalized sigmoid)

In standard PyTorch / torch.compile, this pipeline is split across 3 separate
kernel launches with intermediate global memory roundtrips. This Triton kernel
fuses the entire pipeline into a single pass:
- Logits and bias are loaded into SRAM/registers once.
- Scores and Top-K extraction are computed in registers via iterative argmax.
- Raw logits are gathered directly from registers and activated via sigmoid.
- Only the final (batch, k) IDs and weights are stored to global memory.
"""

from typing import Optional, Tuple
import torch

try:
    import triton
    import triton.language as tl
    IS_TRITON_AVAILABLE = True
except ImportError:
    IS_TRITON_AVAILABLE = False


def is_kolibri_router_available() -> bool:
    """Check if the fused Triton Kolibri router is available in the current environment."""
    return IS_TRITON_AVAILABLE and torch.cuda.is_available()


if IS_TRITON_AVAILABLE:

    @triton.jit
    def _fused_kolibri_router_kernel(
        X_ptr,
        Bias_ptr,
        TopK_Ids_ptr,
        TopK_Weights_ptr,
        stride_xb,
        stride_xe,
        stride_ob,
        stride_ok,
        E: tl.constexpr,
        K: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        NORM_PROB: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < E

        # 1. Load input logits into registers
        x_ptrs = X_ptr + pid * stride_xb + cols * stride_xe
        x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)

        # 2. Add bias if present
        if HAS_BIAS:
            bias = tl.load(Bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            scores = tl.where(mask, x_fp32 + bias, -float("inf"))
        else:
            scores = tl.where(mask, x_fp32, -float("inf"))

        # 3. Iterative Top-K extraction directly in registers
        sum_weight = 0.0
        for k_idx in range(K):
            max_idx = tl.argmax(scores, axis=0)

            # Extract selected raw logit from registers without DRAM re-load
            selected_x = tl.sum(tl.where(cols == max_idx, x_fp32, 0.0), axis=0)
            weight = tl.sigmoid(selected_x)

            if NORM_PROB:
                sum_weight += weight

            # Store selected expert ID and weight
            tl.store(TopK_Ids_ptr + pid * stride_ob + k_idx * stride_ok, max_idx.to(tl.int32))
            tl.store(TopK_Weights_ptr + pid * stride_ob + k_idx * stride_ok, weight)

            # Mask out selected expert for subsequent iterations
            scores = tl.where(cols == max_idx, -float("inf"), scores)

        # Optional probability normalization (if required by checkpoint)
        if NORM_PROB:
            inv_sum = 1.0 / (sum_weight + 1e-20)
            for k_idx in range(K):
                out_ptr = TopK_Weights_ptr + pid * stride_ob + k_idx * stride_ok
                w = tl.load(out_ptr)
                tl.store(out_ptr, w * inv_sum)
else:
    _fused_kolibri_router_kernel = None


def fused_kolibri_router(
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    k: int = 6,
    norm_topk_prob: bool = False,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton entrypoint for Kolibri 1 router scoring & Top-K selection.

    Args:
        x: Input router logits of shape [batch_size, num_experts].
        bias: Optional routing bias of shape [num_experts].
        k: Number of experts per token (default: 6).
        norm_topk_prob: Whether to normalize top-k weights to sum to 1.
        output_dtype: Output dtype for weights (defaults to x.dtype).

    Returns:
        topk_ids: Tensor of shape [batch_size, k] with dtype int32.
        topk_weights: Tensor of shape [batch_size, k] with output_dtype.
    """
    if not IS_TRITON_AVAILABLE:
        raise RuntimeError("Triton is required to run fused_kolibri_router, but it is not available.")
    assert x.is_cuda, "Inputs must be on CUDA"
    b, e = x.shape
    assert e <= 512, f"Kernel currently supports up to 512 experts (got {e})"

    if bias is not None:
        assert bias.is_cuda, "Bias must be on CUDA"
        assert bias.ndim == 1 and bias.shape[0] == e, f"Bias shape must match num_experts ({e})"
        bias = bias.contiguous()

    x = x.contiguous()

    if output_dtype is None:
        output_dtype = x.dtype

    topk_ids = torch.empty((b, k), device=x.device, dtype=torch.int32)
    topk_weights = torch.empty((b, k), device=x.device, dtype=output_dtype)

    grid = (b,)
    _fused_kolibri_router_kernel[grid](
        x,
        bias if bias is not None else x,
        topk_ids,
        topk_weights,
        x.stride(0),
        x.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        E=e,
        K=k,
        BLOCK_SIZE=512,
        HAS_BIAS=(bias is not None),
        NORM_PROB=norm_topk_prob,
        num_warps=2,
    )

    return topk_ids, topk_weights
