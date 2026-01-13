#!/bin/bash

# 模型配置数组: "Name d_model d_ff num_layers num_heads"
CONFIGS=(
    "small 768 3072 12 12"
    "medium 1024 4096 24 16"
    "large 1280 5120 36 20"
    "xl 1600 6400 48 25"
    "2.7B 2560 10240 32 32"
)

WARMUP=5
ITERATIONS=10

echo "开始运行基准测试..."
echo "Warmup steps: $WARMUP"
echo "Measurement steps: $ITERATIONS"
echo "---------------------------------------------------"

for config in "${CONFIGS[@]}"; do
    read -r NAME D_MODEL D_FF NUM_LAYERS NUM_HEADS <<< "$config"
    
    echo "Running benchmark for size: $NAME"
    echo "  (d_model=$D_MODEL, d_ff=$D_FF, layers=$NUM_LAYERS, heads=$NUM_HEADS)"
    
    uv run python cs336_systems/benchmarking.py \
        --d_model "$D_MODEL" \
        --d_ff "$D_FF" \
        --num_layers "$NUM_LAYERS" \
        --num_heads "$NUM_HEADS" \
        --warmup_iterations "$WARMUP" \
        --benchmark_iterations "$ITERATIONS" \
        --batch_size 2 \
        --context_length 256 \
        --dtype "bfloat16" \
        --forward_or_backward "both"
        
    echo "---------------------------------------------------"
done

echo "---------------------------------------------------"
echo "开始 Warmup 影响测试 (使用 Medium 模型)..."

# Medium 模型配置
D_MODEL=1024
D_FF=4096
NUM_LAYERS=24
NUM_HEADS=16
BATCH_SIZE=2
CONTEXT_LENGTH=256
ITERATIONS=10

for WARMUP in 0 1 2 5; do
    echo "Running benchmark with WARMUP_STEPS=$WARMUP"
    uv run python cs336_systems/benchmarking.py \
        --d_model "$D_MODEL" \
        --d_ff "$D_FF" \
        --num_layers "$NUM_LAYERS" \
        --num_heads "$NUM_HEADS" \
        --warmup_iterations "$WARMUP" \
        --benchmark_iterations "$ITERATIONS" \
        --batch_size "$BATCH_SIZE" \
        --context_length "$CONTEXT_LENGTH" \
        --dtype "bfloat16" \
        --forward_or_backward "both"
    echo "---------------------------------------------------"
done

# Deliverable 问题 (b):
# How long does a forward pass take? How about a backward pass? 
# Do you see high variability across measurements, or is the standard deviation small?

# 你的回答（基于 NVIDIA RTX 4090 D, context_length=256, batch_size=2, bfloat16）：
#
# 前向传播耗时 (Forward Pass Time):
# 随着模型尺寸增大，推理模式（eval）下的前向传播耗时从 small 模型的约 9.4ms (0.0094s) 增加到 2.7B 模型的约 38.4ms (0.0384s)。
# 值得注意的是，xl 模型 (48层) 的前向耗时 (35.6ms) 与 2.7B 模型 (32层, 但更宽) 的耗时 (38.4ms) 相当接近。
# 训练模式下的前向传播通常比推理模式稍慢（约慢 10-20%），因为需要构建计算图。
#
# 反向传播耗时 (Backward Pass Time):
# 反向传播耗时通常是前向传播耗时的 2 到 3 倍左右。
# 例如，small 模型的反向传播约为 16.9ms，而 2.7B 模型增加到了约 126.0ms。
#
# 波动性 (Variability/Standard Deviation):
# 前向传播 (eval) 的标准差非常小 (std < 1ms)，表明测量结果非常稳定。
# 反向传播的标准差相对较大 (std 在 15ms-20ms 左右)，这可能是由于显存分配、垃圾回收或者 GPU 动态频率调整导致的波动。
# 总体而言，测量结果具有较好的一致性。

# Deliverable 问题 (c):
# One caveat of benchmarking is not performing the warm-up steps. 
# Repeat your analysis without the warm-up steps. How does this affect your results? 
# Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. 
# Why might the result still be different?

# 你的回答：
#
# 没有 Warm-up (0 steps) 的影响：
# 实际测试结果显示，不进行 warm-up 时，推理模式（eval）下的 Forward Pass 平均耗时从正常的约 18ms 激增至 41.4ms，且标准差极高（达到 69.5ms）。
# 这种巨大的波动和性能损失主要是由于第一次运行时必须进行的初始化开销，包括 CUDA Context 初始化、内存分配器的冷启动、以及 PyTorch 内部算子的首次调用开销（如 Autotuning）。
#
# 少量 Warm-up (1-2 steps) 的影响：
# 在本实验中，即使只进行 1 步 warm-up，Forward Pass 的性能也迅速恢复正常（约 18.4ms），标准差降至 0.2ms，与 5 步 warm-up 的结果（18.1ms）几乎一致。
# 这表明在当前简单的 Benchmark 场景下，PyTorch 和 GPU 能够非常快地进入稳态。
# 尽管如此，为了确保在更复杂的模型或不同的硬件/软件栈上也能获得可靠结果，通常仍建议保留 5 步以上的 warm-up。
