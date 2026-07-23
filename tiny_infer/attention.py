"""多头因果自注意力的最小实现。"""

import math
from typing import NamedTuple

import torch
from torch import nn


class AttentionOutput(NamedTuple):
    """保存注意力各阶段结果，避免使用难以辨认的数字下标。"""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    scores: torch.Tensor
    weights: torch.Tensor
    context: torch.Tensor
    output: torch.Tensor


class SelfAttention(nn.Module):
    """执行 Q/K/V 投影、因果注意力和输出投影。"""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 一次线性层同时生成 Q、K、V。
        self.qkv_projection = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=False,
        )
        self.output_projection = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, H] -> [B, N, S, D]，其中 H=N*D。"""
        batch_size, sequence_length, _ = x.shape

        # 先拆出 head 维度，再把 head 放到序列前面。
        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        return x.transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, S, D] -> [B, S, H]。"""
        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, sequence_length, self.hidden_size)

    def forward(self, x: torch.Tensor) -> AttentionOutput:
        """输入 [B, S, H]，返回注意力各阶段的张量。"""
        # 1. QKV 投影：[B, S, H] -> [B, S, 3H]。
        qkv = self.qkv_projection(x)

        # 2. 切分 Q、K、V：三个张量都是 [B, S, H]。
        q, k, v = qkv.chunk(3, dim=-1)

        # 3. 拆分多头：三个张量都是 [B, N, S, D]。
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 4. 相似度：每个 head 内 [S, D] @ [D, S] -> [S, S]。
        scores = torch.matmul(q, k.transpose(-2, -1))

        # 5. 缩放分数，避免 softmax 过于尖锐。
        scores = scores / math.sqrt(self.head_dim)

        # 6. 创建 [S, S] 的上三角因果遮罩，禁止读取未来 token。
        causal_mask = torch.triu(
            torch.ones(
                scores.size(-2),
                scores.size(-1),
                dtype=torch.bool,
                device=scores.device,
            ),
            diagonal=1,
        )

        # 7. 被遮罩的位置填入 -inf，softmax 后权重为 0。
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # 8. 分数转为注意力权重，形状仍是 [B, N, S, S]。
        weights = torch.softmax(scores, dim=-1)

        # 9. 加权求和：[S, S] @ [S, D] -> [S, D]。
        context = torch.matmul(weights, v)

        # 10. 合并多个 head：[B, N, S, D] -> [B, S, H]。
        merged_context = self.merge_heads(context)

        # 11. 输出投影，形状保持 [B, S, H]。
        output = self.output_projection(merged_context)

        return AttentionOutput(
            q=q,
            k=k,
            v=v,
            scores=scores,
            weights=weights,
            context=context,
            output=output,
        )
