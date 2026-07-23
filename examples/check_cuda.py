"""检查 PyTorch 是否可以使用 NVIDIA GPU。"""

import torch


def main() -> None:
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 无法访问 NVIDIA GPU")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))

    # 用矩阵乘法验证 GPU 能执行实际张量计算。
    x = torch.randn(2048, 2048, device="cuda")
    y = torch.randn(2048, 2048, device="cuda")
    z = x @ y

    # CUDA 运算可能异步执行，同步后再读取结果信息。
    torch.cuda.synchronize()

    print("Result shape:", tuple(z.shape))
    print("Result device:", z.device)


if __name__ == "__main__":
    main()
