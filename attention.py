import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """自注意力的 Q/K/V 投影与多头拆分。"""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_projection = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=False,
        )

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv_projection(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # K 的最后两个维度从 [S, D] 交换成 [D, S]，
        # 然后每个 Query token 与所有 Key token 计算匹配分数。
        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        # 防止点积结果随着 head_dim 增大而过大。
        scores = scores / math.sqrt(self.head_dim)

        # 创建一个右上角为 True 的遮罩，表示未来 token。
        causal_mask = torch.triu(
            torch.ones(
                scores.size(-2),
                scores.size(-1),
                dtype=torch.bool,
                device=scores.device,
            ),
            diagonal=1,
        )

        # 把未来位置的分数改成负无穷。
        scores = scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        weights = torch.softmax(scores, dim=-1)

        # 按照注意力权重，混合各个 token 的 V。
        context = torch.matmul(weights, v)

        return q, k, v, scores, weights, context