import torch


def main() -> None:
    # 推理通常会反复执行大量张量运算，因此第一步是确认：
    # 1. 当前安装的 PyTorch 版本；
    # 2. PyTorch 是否带有 CUDA 运行时；
    # 3. 当前进程是否真的能访问 NVIDIA GPU。
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        # 后面的张量会显式放到 cuda 上。没有 GPU 时提前报错，
        # 比等到创建张量时才看到较难理解的错误更容易定位问题。
        raise RuntimeError("PyTorch 无法访问 NVIDIA GPU")

    # 设备编号 0 表示当前可见的第一张 GPU。
    print("GPU:", torch.cuda.get_device_name(0))
    # Compute capability 是 NVIDIA GPU 的硬件能力版本，例如 (8, 6)。
    print("Compute capability:", torch.cuda.get_device_capability(0))

    # device="cuda" 会把数据直接创建在 GPU 显存中，避免之后再拷贝一次。
    x = torch.randn(2048, 2048, device="cuda")
    y = torch.randn(2048, 2048, device="cuda")

    # 矩阵乘法是神经网络中最常见的计算之一，线性层和注意力层都大量使用它。
    z = x @ y

    # CUDA 运算通常是异步提交的。同步后，才能确保矩阵乘法已经真正完成，
    # 也能让潜在的 GPU 错误在这里暴露出来。
    torch.cuda.synchronize()

    # 这些信息用于确认结果的形状和存放位置符合预期。
    print("Result shape:", tuple(z.shape))
    print("Result device:", z.device)


if __name__ == "__main__":
    main()
