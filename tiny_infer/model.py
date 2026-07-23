"""把已有组件组合成一个最小的完整语言模型。"""

import torch
from torch import nn

from .rms_norm import RMSNorm
from .transformer_block import TransformerBlock


class TinyDecoderModel(nn.Module):
    """接收 token 编号，输出每个位置的候选 token 分数。"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size

        # 把每个 token 编号查成一组 hidden_size 维的数字。
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )

        # 连续放置多个 TransformerBlock。
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                )
                for _ in range(num_layers)
            ]
        )

        # 所有层处理结束后，再稳定一次 token 的数值范围。
        self.final_norm = RMSNorm(hidden_size)

        # 创建一个 [vocab_size, hidden_size] 的可学习权重矩阵，
        # 把每个 token 的隐藏向量转换成词表分数。
        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """输入 [B, S]，输出 [B, S, V]。"""

        # token 编号：[B, S]
        # token 数字表示：[B, S, H]
        x = self.token_embedding(token_ids)

        # 依次经过每一个 TransformerBlock。
        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        # 调用 lm_head 的 Linear.forward：
        # [B, S, H] -> [B, S, vocab_size]。
        logits = self.lm_head(x)

        return logits
