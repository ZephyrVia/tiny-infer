"""观察线性层如何改变每个 token 的内部维度。"""

import torch
from torch import nn


def main() -> None:
    # 优先使用 GPU；没有 GPU 时自动改用 CPU。
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2 条序列，每条有 4 个 token，
    # 每个 token 目前由 64 个数字表示。
    x = torch.randn(
        2,
        4,
        64,
        device=device,
    )

    # 把每个 token 的 64 个数字转换成 128 个数字。
    projection = nn.Linear(
        in_features=64,
        out_features=128,
        bias=False,
    ).to(device)

    # 进入推理模式；Linear 本身没有 dropout，但这是推理时的常见写法。
    projection.eval()

    # 推理阶段不需要保存梯度，可以节省内存。
    with torch.inference_mode():
        output = projection(x)

    print("device:", device)
    print("input shape:", x.shape)
    print("weight shape:", projection.weight.shape)
    print("output shape:", output.shape)

    # 检查 Linear 只改变最后一维，前面的批次和序列维度保持不变。
    assert x.shape == (2, 4, 64)
    assert projection.weight.shape == (128, 64)
    assert output.shape == (2, 4, 128)

    print("linear projection test passed")


if __name__ == "__main__":
    main()
