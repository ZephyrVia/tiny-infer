"""演示一次 RMSNorm 推理，并检查输入输出是否符合预期。"""

import torch

from tiny_infer.rms_norm import RMSNorm


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 用简单数字观察 RMSNorm 的作用。
    toy_x = torch.tensor(
        [
            [
                [3.0, 4.0],
                [6.0, 8.0],
            ]
        ],
        device=device,
    )

    toy_norm = RMSNorm(hidden_size=2).to(device)
    toy_norm.eval()

    with torch.inference_mode():
        toy_output = toy_norm(toy_x)

    print("toy input:")
    print(toy_x)

    print("toy output:")
    print(toy_output)

    # 准备一批模拟的 token 隐藏状态，形状为 [B, S, H]。
    x = torch.randn(
        2,
        4,
        64,
        device=device,
    )

    # 创建 RMSNorm；它会沿每个 token 的 hidden 维度进行调整。
    norm = RMSNorm(hidden_size=64).to(device)
    norm.eval()

    # 推理时不记录梯度，直接得到调整后的结果。
    with torch.inference_mode():
        output = norm(x)

    print("device:", device)
    print("input shape:", x.shape)
    print("weight shape:", norm.weight.shape)
    print("output shape:", output.shape)

    # 检查输入输出形状是否保持不变。
    assert output.shape == x.shape
    assert norm.weight.shape == (64,)

    # 检查每个 token 是否已经调整到标准尺度。
    output_rms = torch.sqrt(
        output.pow(2).mean(dim=-1)
    )

    torch.testing.assert_close(
        output_rms,
        torch.ones_like(output_rms),
        atol=1e-5,
        rtol=1e-5,
    )

    print("output rms:")
    print(output_rms)
    print("RMSNorm inference test passed")


if __name__ == "__main__":
    main()
