"""运行一次注意力前向推理，并打印关键张量。"""

import torch

from tiny_infer.attention import SelfAttention


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 2
    sequence_length = 4
    hidden_size = 64
    num_heads = 4

    x = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        device=device,
    )

    attention = SelfAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    ).to(device)
    attention.eval()

    # inference_mode 关闭梯度记录，模拟真正的推理流程。
    with torch.inference_mode():
        result = attention(x)

    print("device:", device)
    print("input shape:", x.shape)
    print("q shape:", result.q.shape)
    print("k shape:", result.k.shape)
    print("v shape:", result.v.shape)
    print("scores shape:", result.scores.shape)
    print("weights shape:", result.weights.shape)
    print("context shape:", result.context.shape)
    print("output shape:", result.output.shape)

    expected_head_shape = (2, 4, 4, 16)
    assert result.q.shape == expected_head_shape
    assert result.k.shape == expected_head_shape
    assert result.v.shape == expected_head_shape
    assert result.scores.shape == (2, 4, 4, 4)
    assert result.weights.shape == (2, 4, 4, 4)
    assert result.context.shape == expected_head_shape
    assert result.output.shape == (2, 4, 64)

    # 第一个 token 不能看到后面三个 token。
    assert torch.isneginf(result.scores[0, 0, 0, 1:]).all()
    assert (result.weights[0, 0, 0, 1:] == 0).all()

    # 每一行注意力权重都应当和为 1。
    torch.testing.assert_close(
        result.weights[0, 0].sum(dim=-1),
        torch.ones(sequence_length, device=device),
    )

    print("attention inference test passed")


if __name__ == "__main__":
    main()
