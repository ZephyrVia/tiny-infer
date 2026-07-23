"""用最直接的方式逐个生成 token。"""

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

    # 暂时用数字代替真实文字 token。
    token_ids = torch.tensor(
        [[12, 38, 91, 7]],
        dtype=torch.long,
        device=device,
    )

    max_new_tokens = 5

    print("initial token ids:")
    print(token_ids)

    with torch.inference_mode():
        for step in range(max_new_tokens):
            # 对当前完整序列进行一次模型计算。
            logits = model(token_ids)

            # 只看最后一个位置给出的候选分数。
            next_token_logits = logits[:, -1, :]

            # 选择分数最高的 token。
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            # 把新 token 拼到序列末尾。
            token_ids = torch.cat(
                [token_ids, next_token],
                dim=1,
            )

            print(
                f"step {step + 1}:",
                "new token =",
                next_token.item(),
            )
            print("current token ids:", token_ids)

    assert token_ids.shape == (1, 4 + max_new_tokens)

    print("generation demo passed")


if __name__ == "__main__":
    main()
