"""运行一次完整 Transformer 层的前向推理。"""

import torch

from tiny_infer import TransformerBlock


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 2 条序列，每条 4 个 token，
    # 每个 token 由 64 个数字表示。
    x = torch.randn(
        2,
        4,
        64,
        device=device,
    )

    block = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
    ).to(device)

    block.eval()

    with torch.inference_mode():
        output = block(x)

    print("device:", device)
    print("input shape:", x.shape)
    print("output shape:", output.shape)

    assert output.shape == x.shape

    # 完整处理后，内容应该发生变化。
    assert not torch.equal(output, x)

    print("transformer block inference test passed")


if __name__ == "__main__":
    main()