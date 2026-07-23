"""运行一次最小完整模型的前向推理。"""

import torch

from tiny_infer.model import TinyDecoderModel


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocab_size = 1000

    model = TinyDecoderModel(
        vocab_size=vocab_size,
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        intermediate_size=128,
    ).to(device)

    model.eval()

    # 这里的数字暂时代表 token 编号。
    #
    # 2 条序列，每条序列有 4 个 token。
    token_ids = torch.tensor(
        [
            [12, 38, 91, 7],
            [26, 15, 80, 3],
        ],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        # 先单独查看 Embedding 的输出：每个 token 编号变成一个 hidden 向量。
        embeddings = model.token_embedding(token_ids)
        logits = model(token_ids)

    print("device:", device)
    print("token ids shape:", token_ids.shape)
    print("token embedding table shape:", model.token_embedding.weight.shape)
    print("embedding output shape:", embeddings.shape)
    print("lm head weight shape:", model.lm_head.weight.shape)
    print("logits shape:", logits.shape)

    # 查看第 1 条序列最后一个位置的最高分候选 token。
    top_scores, top_token_ids = torch.topk(logits[0, -1], k=5)
    print("top candidate token ids:", top_token_ids)
    print("top candidate scores:", top_scores)

    assert token_ids.shape == (2, 4)
    assert logits.shape == (2, 4, vocab_size)

    print("tiny decoder model inference test passed")


if __name__ == "__main__":
    main()
