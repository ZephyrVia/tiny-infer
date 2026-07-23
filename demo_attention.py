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

    q, k, v, scores = attention(x)

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

    

    print("Q/K/V shape test passed")
    


if __name__ == "__main__":
    main()