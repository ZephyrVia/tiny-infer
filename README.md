# tiny-infer

一个用于学习 PyTorch 推理和 Transformer 注意力机制的最小项目。

## 目录结构

```text
tiny-infer/
├── tiny_infer/              # 核心源码
│   ├── __init__.py
│   └── attention.py         # SelfAttention 实现
├── examples/                # 可直接运行的学习示例
│   ├── attention_demo.py    # 注意力前向推理和形状检查
│   └── check_cuda.py        # CUDA 环境检查
├── requirements.txt
└── .gitignore
```

## 如何运行

所有命令都要在项目根目录 `tiny-infer/` 下执行。

`-m` 表示“按模块运行”。例如：

```bash
python -m examples.attention_demo
```

它会把 `examples/attention_demo.py` 作为 Python 模块执行，
同时保证可以正确导入同级的 `tiny_infer` 包。

运行注意力推理示例：

```bash
python -m examples.attention_demo
```

运行 CUDA 环境检查：

```bash
python -m examples.check_cuda
```

如果当前环境没有 NVIDIA GPU，CUDA 检查会明确报错；注意力示例仍可在 CPU 上运行。

## 注意力流程

`tiny_infer/attention.py` 中的 `SelfAttention` 按以下顺序执行：

```text
x
  -> Q/K/V 投影
  -> 拆分多个 head
  -> Q @ K.T
  -> 缩放和因果遮罩
  -> softmax 得到权重
  -> 权重 @ V
  -> 合并 head
  -> 输出投影
```

`forward()` 返回 `AttentionOutput`，可以使用 `result.q`、
`result.weights`、`result.context` 和 `result.output` 等名称查看结果。
