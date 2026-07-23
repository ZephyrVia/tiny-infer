"""把注意力和 token 内部处理组合成一个完整层。"""

import torch
from torch import nn

from .attention import SelfAttention
from .feed_forward import FeedForward
from .rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    """先让 token 交流，再让每个 token 单独整理信息。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
    ):
        super().__init__()

        # Attention 之前，先稳定每个 token 的数值范围。
        self.attention_norm = RMSNorm(hidden_size)

        # 让每个 token 从自己和前面的 token 中读取信息。
        self.attention = SelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        # FeedForward 之前，再稳定一次数值范围。
        self.feed_forward_norm = RMSNorm(hidden_size)

        # 每个 token 单独消化和整理自己的信息。
        self.feed_forward = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入和输出形状都为 [B, S, H]。"""

        # 先让 token 之间交流。
        attention_input = self.attention_norm(x)

        # Attention 会返回多个中间结果；这里只取最终的 [B, S, H] 输出。
        attention_output = self.attention(attention_input).output

        # 保留原来的信息，同时加入 Attention 得到的新信息。
        x = x + attention_output

        # 再让每个 token 单独整理自己的信息。
        feed_forward_input = self.feed_forward_norm(x)
        feed_forward_output = self.feed_forward(feed_forward_input)

        # 继续保留已有信息，并加入整理后的结果。
        return x + feed_forward_output
