# Benchmark 模块

此文件夹包含性能基准测试和分析工具。

## 文件说明

### 1. benchmarking.py
基本的端到端基准测试脚本，用于测量模型的前向和反向传播时间。

**用法：**
```bash
uv run python cs336_systems/benchmark/benchmarking.py \
    --d_model 768 \
    --d_ff 3072 \
    --num_layers 12 \
    --num_heads 12 \
    --warmup_iterations 10 \
    --benchmark_iterations 100 \
    --batch_size 4 \
    --context_length 1024 \
    --dtype bfloat16 \
    --forward_or_backward both
```

### 2. profile_model.py
使用 NVTX 标记的性能分析脚本，配合 Nsight Systems 使用。

**用法：**
```bash
nsys profile -t cuda,nvtx,osrt -o output \
    uv run python cs336_systems/benchmark/profile_model.py \
        --d_model 768 \
        --d_ff 3072 \
        --num_layers 12 \
        --num_heads 12 \
        --context_length 1024 \
        --batch_size 2 \
        --dtype bfloat16
```

### 3. benchmarking_mixed_precision.py
包含混合精度训练相关的模型定义和分析。

### 4. mixed_precision_accumulation.py
演示混合精度累加的数值特性。

**用法：**
```bash
uv run python cs336_systems/benchmark/mixed_precision_accumulation.py
```

## 快速开始

### 运行完整基准测试
```bash
bash scripts/run_py_benchmark.sh
```

### 运行 Nsight Systems 性能分析
```bash
bash scripts/run_nsys_benchmarks.sh
```

## 注意事项

- 所有脚本都需要 CUDA GPU 才能运行
- 建议使用 `bfloat16` 数据类型以获得更好的性能
- 基准测试前会自动进行预热步骤以确保测量准确性
