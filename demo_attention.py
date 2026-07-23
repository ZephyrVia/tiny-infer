import torch

from attention import SelfAttention


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    q, k, v, scores, weights, context = attention(x)

    print("scores shape:", scores.shape)

    assert scores.shape == (2, 4, 4, 4)

    print("input shape:", x.shape)
    print("q shape:", q.shape)
    print("k shape:", k.shape)
    print("v shape:", v.shape)

    expected_shape = (2, 4, 4, 16)

    assert q.shape == expected_shape
    assert k.shape == expected_shape
    assert v.shape == expected_shape

    print("scores for batch 0, head 0:")
    print(scores[0, 0])

    assert torch.isneginf(scores[0, 0, 0, 1])
    assert torch.isneginf(scores[0, 0, 0, 2])
    assert torch.isneginf(scores[0, 0, 0, 3])

    print("weights for batch 0, head 0:")
    print(weights[0, 0])

    print("weight row sums:")
    print(weights[0, 0].sum(dim=-1))

    assert weights.shape == (2, 4, 4, 4)

    expected_sums = torch.ones(
        4,
        device=weights.device,
    )

    torch.testing.assert_close(
        weights[0, 0].sum(dim=-1),
        expected_sums,
    )

    assert weights[0, 0, 0, 1] == 0
    assert weights[0, 0, 0, 2] == 0
    assert weights[0, 0, 0, 3] == 0

    print("context shape:", context.shape)

    assert context.shape == (2, 4, 4, 16)

    print("Q/K/V shape test passed")
    


if __name__ == "__main__":
    main()
