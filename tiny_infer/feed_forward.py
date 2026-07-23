"""让每个 token 单独整理自己的信息。"""

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """先展开 token 的内部表示，再筛选并压回原来的大小。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()

        # 生成控制信号，帮助模型决定哪些信息应该通过。
        # [B, S, H] -> [B, S, I]。
        self.gate_projection = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        # 把每个 token 的信息展开到更大的中间空间。
        # [B, S, H] -> [B, S, I]。
        self.up_projection = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )

        # 把处理后的信息压回原来的 hidden_size。
        # [B, S, I] -> [B, S, H]。
        self.down_projection = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入和输出形状都是 [B, S, H]。"""

        # 生成控制信号，决定展开后的哪些信息应该保留。
        gate = F.silu(self.gate_projection(x))

        # 生成展开后的 token 信息。
        expanded = self.up_projection(x)

        # 用控制信号筛选展开后的信息。
        filtered = gate * expanded

        # 压回原来的 hidden_size。
        return self.down_projection(filtered)
