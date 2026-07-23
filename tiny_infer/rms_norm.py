"""RMSNorm：把每个 token 的数值范围调整得更稳定。"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """让每个 token 的整体数值大小保持稳定。"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.eps = eps

        # 每个 hidden 维度都有一个可学习的缩放参数，
        # 让模型可以在归一化后继续微调不同维度的大小。
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """调整每个 token 的数值范围，输入输出形状都是 [B, S, H]。"""

        # 计算每个 token 整体有多大。
        # [B, S, H] -> [B, S, 1]，保留最后一维方便后面调整原输入。
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)

        # 把每个 token 调整到稳定的数值范围。
        normalized = x * torch.rsqrt(mean_square + self.eps)

        # 用可学习权重做最后的微调；[H] 会自动扩展到 [B, S, H]。
        return normalized * self.weight
