"""运行一次 token 内部信息处理。"""

import torch

from tiny_infer.feed_forward import FeedForward


def main() -> None:
    # 优先使用 GPU；没有 GPU 时自动改用 CPU。
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2 条序列，每条 4 个 token，每个 token 有 64 个数字。
    x = torch.randn(
        2,
        4,
        64,
        device=device,
    )

    # 创建一个把 hidden_size=64 展开到 intermediate_size=128 的模块。
    feed_forward = FeedForward(
        hidden_size=64,
        intermediate_size=128,
    ).to(device)

    # 切换到推理状态，并关闭梯度记录。
    feed_forward.eval()

    with torch.inference_mode():
        output = feed_forward(x)

    print("device:", device)
    print("input shape:", x.shape)
    print(
        "gate weight shape:",
        feed_forward.gate_projection.weight.shape,
    )
    print(
        "up weight shape:",
        feed_forward.up_projection.weight.shape,
    )
    print(
        "down weight shape:",
        feed_forward.down_projection.weight.shape,
    )
    print("output shape:", output.shape)

    # FeedForward 处理 token 内部信息，但最终要恢复原来的形状。
    assert output.shape == x.shape
    assert feed_forward.gate_projection.weight.shape == (128, 64)
    assert feed_forward.up_projection.weight.shape == (128, 64)
    assert feed_forward.down_projection.weight.shape == (64, 128)

    print("feed-forward inference test passed")


if __name__ == "__main__":
    main()
