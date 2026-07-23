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
        """把最后一个隐藏维度拆成多个注意力头。

        以本项目的参数为例：

            [B, S, H] = [2, 4, 64]
            [B, S, N, D] = [2, 4, 4, 16]
            [B, N, S, D] = [2, 4, 4, 16]

        B 是 batch_size，S 是序列长度，N 是头数，D 是每个头的维度。
        """
        batch_size, sequence_length, _ = x.shape

        # 只改变观察张量的方式，不改变数值：
        # [B, S, H] -> [B, S, N, D]，其中 H = N * D。
        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        # 把 head 维度提前，便于每个 head 独立进行矩阵乘法：
        # [B, S, N, D] -> [B, N, S, D]。
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """执行多头自注意力的主要矩阵运算。

        本项目的示例参数是 B=2、S=4、H=64、N=4、D=16，
        因此下面注释中的具体形状分别对应 [2, 4, 64] 等张量。
        """

        # 第 0 步：输入隐藏状态 x 是一个三维张量，不是单个矩阵。
        # 如果固定第 b 条样本，x[b] 才是一个二维矩阵 [S, H] = [4, 64]；
        # 如果继续固定第 s 个 token，x[b, s] 就是一个一维向量 [H] = [64]。
        #
        #     x = [B, S, H] = [2, 4, 64]
        #
        # 表示 2 条样本、每条 4 个 token、每个 token 用 64 个数表示。
        # 第 1 步：一次线性层同时生成 Q、K、V。
        #
        #     x         [B, S, H]      [2, 4, 64]
        #     线性层权重 [3H, H]        [192, 64]
        #     数学上的   W^T [H, 3H]    [64, 192]
        #     qkv       [B, S, 3H]     [2, 4, 192]
        #
        # 最后一维 192 依次存放 Q(64)、K(64)、V(64)。
        qkv = self.qkv_projection(x)

        # 第 2 步：切出三个三维张量，各自形状为 [B, S, H]：
        #
        #     q = [2, 4, 64]
        #     k = [2, 4, 64]
        #     v = [2, 4, 64]
        #
        # Q 表示“我想查询什么”，K 表示“我有什么特征可被匹配”，
        # V 表示“匹配后要取出的内容”。
        q, k, v = qkv.chunk(3, dim=-1)

        # 第 3 步：分别把 Q、K、V 的隐藏维度拆成多个 head。
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # 多头拆分完成后，三个张量都是：
        #
        #     Q, K, V = [B, N, S, D] = [2, 4, 4, 16]
        #
        # 现在有 4 个 head，每个 head 只处理 16 维特征。

        # 第 4 步：计算 Query 和 Key 的相似度。
        # Q、K 是四维张量；固定一个 batch 和一个 head 后，
        # q[0, 0]、k[0, 0] 才是实际参与运算的二维矩阵：
        #
        #     q[0, 0]             [S, D] = [4, 16]
        #     k[0, 0].transpose() [D, S] = [16, 4]
        #     q[0, 0] @ k[0, 0].T [S, S] = [4, 4]
        #
        # 下面的 torch.matmul 会对所有 B*N 个 batch-head 矩阵同时计算，
        # 所以整体结果 scores 是四维张量，而不是单个二维矩阵。
        # K 的最后两个维度从 [S, D] 交换成 [D, S]：
        #
        #     Q             [B, N, S, D] = [2, 4, 4, 16]
        #     K.transpose()  [B, N, D, S] = [2, 4, 16, 4]
        #     scores        [B, N, S, S] = [2, 4, 4, 4]
        #
        # scores[b, n] 是一个 [S, S] 矩阵；其中每一行表示一个 Query token
        # 对 4 个 Key token 的匹配分数。
        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        # 第 5 步：缩放分数矩阵。
        #
        #     scores = scores / sqrt(D)
        #            = [2, 4, 4, 4] / sqrt(16)
        #            = [2, 4, 4, 4]
        #
        # 形状不变，只把数值缩小，防止 D 变大时 softmax 过于尖锐。
        scores = scores / math.sqrt(self.head_dim)

        # 第 6 步：创建因果遮罩 causal_mask，它本身是一个二维矩阵，
        # 形状为 [S, S] = [4, 4]。
        # True 表示当前位置不能看到未来 token。对于 S=4，它是：
        #
        #     [[False, True,  True,  True ],
        #      [False, False, True,  True ],
        #      [False, False, False, True ],
        #      [False, False, False, False]]
        #
        # 它会广播到 scores 的 [B, N, S, S]，每个 batch、每个 head 共用一份。
        causal_mask = torch.triu(
            torch.ones(
                scores.size(-2),
                scores.size(-1),
                dtype=torch.bool,
                device=scores.device,
            ),
            diagonal=1,
        )

        # 第 7 步：把不能看的位置改为负无穷：
        #
        #     scores = [2, 4, 4, 4]
        #     未来位置 = -inf
        #
        # softmax(-inf) 会变成 0，因此当前 token 不会从未来 token 获取信息。
        scores = scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        # 第 8 步：对最后一个维度做 softmax，把每个 scores[b, n] 矩阵
        # 的每一行分数变成概率权重：
        #
        #     weights = softmax(scores, dim=-1)
        #     weights  [B, N, S, S] = [2, 4, 4, 4]
        #
        # 每一行权重之和为 1；因果遮罩对应的位置权重为 0。
        weights = torch.softmax(scores, dim=-1)

        # 第 9 步：对固定的 batch 和 head，实际计算的是两个二维矩阵：
        #
        #     weights[0, 0] [S, S] = [4, 4]
        #     v[0, 0]       [S, D] = [4, 16]
        #     两者相乘                  [S, D] = [4, 16]
        #
        # 对所有 batch-head 同时计算后，得到每个 head 的上下文张量：
        #
        #     weights  [B, N, S, S] = [2, 4, 4, 4]
        #     V        [B, N, S, D] = [2, 4, 4, 16]
        #     context  [B, N, S, D] = [2, 4, 4, 16]
        #
        # 目前示例在这里结束；完整 Transformer 通常还会把 context
        # 转置并合并多个 head，恢复成 [B, S, H] = [2, 4, 64]。
        context = torch.matmul(weights, v)

        return q, k, v, scores, weights, context
