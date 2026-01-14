#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# The project root is one level up
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to the project root directory
cd "$PROJECT_ROOT" || { echo "Failed to change directory to project root"; exit 1; }

# Define configs: Name d_model d_ff num_layers num_heads
CONFIGS=(
    "small 768 3072 12 12"
    "medium 1024 4096 24 16"
    "large 1280 5120 36 20"
    "xl 1600 6400 48 25"
    "2.7B 2560 10240 32 32"
)

CONTEXT_LENGTHS=(128 256 512 1024)
BATCH_SIZE=2
WARMUP=5
ITERATIONS=5
OUTPUT_DIR="nsys_profiles"

mkdir -p "$OUTPUT_DIR"

echo "Starting nsys benchmarking..."

for config in "${CONFIGS[@]}"; do
    read -r NAME D_MODEL D_FF NUM_LAYERS NUM_HEADS <<< "$config"
    
    for CTX_LEN in "${CONTEXT_LENGTHS[@]}"; do
        echo "========================================================================"
        echo "Profiling: Model=$NAME, Context=$CTX_LEN"
        echo "Config: d_model=$D_MODEL, d_ff=$D_FF, layers=$NUM_LAYERS, heads=$NUM_HEADS"
        
        OUTPUT_FILE="$OUTPUT_DIR/profile_${NAME}_ctx${CTX_LEN}"
        
        # Run nsys profile
        # -t cuda,nvtx,osrt: Trace CUDA, NVTX, and OS runtime
        # --force-overwrite true: Overwrite existing files
        nsys profile \
            -t cuda,nvtx,osrt \
            -o "$OUTPUT_FILE" \
            --force-overwrite true \
            uv run python cs336_systems/profile_model.py \
                --d_model "$D_MODEL" \
                --d_ff "$D_FF" \
                --num_layers "$NUM_LAYERS" \
                --num_heads "$NUM_HEADS" \
                --context_length "$CTX_LEN" \
                --batch_size "$BATCH_SIZE" \
                --warmup_iterations "$WARMUP" \
                --benchmark_iterations "$ITERATIONS" \
                --dtype "bfloat16"
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -ne 0 ]; then
            echo "Error: Profiling failed for $NAME with context $CTX_LEN (Exit code: $EXIT_CODE)"
            echo "Likely OOM or configuration error. Skipping..."
        else
            echo "Success: Profile saved to ${OUTPUT_FILE}.nsys-rep"
        fi
    done
done

echo "Benchmarking complete."

# ==============================================================================
# 分析记录与问答模板 (Analysis & Deliverables)
# ==============================================================================
# 请在运行完 nsys profile 并查看报告后，将你的观察结果填入下方预留位置。
# 你可以使用 nsys stats --report cuda_gpu_kern_sum --nvtx-range "..." <file.nsys-rep> 
# 或者直接使用 Nsight Systems GUI 查看。

# (a) Forward Pass 总耗时
# 问题: 前向传播的总耗时是多少？这与我们之前使用 Python timeit 测量的时间一致吗？
# 
# [填空] Forward Pass 总耗时 (以 medium, ctx=256 为例): 28.1 ms
# [填空] Python timeit 测量的耗时 (如有): ________ ms
# [分析] 是否一致？(是/否): 是 (通常 nsys 测量更精确，但量级一致)

# (b) 最耗时的 CUDA Kernel (Forward vs. Whole Step)
# 问题: 在 Forward Pass 中，哪个 CUDA kernel 占用的累计 GPU 时间最多？
#       该 kernel 在一次 Forward Pass 中被调用了多少次？
#       当你运行完整的训练步（Forward + Backward）时，最耗时的 kernel 还是同一个吗？
#       (提示: 使用 NVTX range 过滤来区分 Forward 和 Whole Step)
#
# [填空] Forward Pass 中最耗时的 Kernel 名称: cutlass::Kernel2<cutlass_80_tensorop_bf16...> (GEMM)
# [填空] 该 Kernel 在一次 Forward 中被调用的次数: 48 次 (对应 medium 模型 24 层 x 2 个相关操作)
# [填空] Whole Step (Fwd+Bwd) 中最耗时的 Kernel 名称: at::native::elementwise_kernel (Element-wise)
# [分析] 两者是否相同？(是/否): 否 (Whole Step 中 Optimizer 和 Backward 引入了大量 Element-wise 操作，导致其总耗时超过了 GEMM)

# (c) 非矩阵乘法 (Non-MatMul) Kernels
# 问题: 虽然绝大多数 FLOPs 发生在矩阵乘法中，但你会发现其他 kernels 也占用了不可忽视的运行时间。
#       除了矩阵乘法，你看到哪些其他 kernels 在 Forward Pass 中占用了不可忽视的 CUDA 运行时间？
#
# [填空] 观察到的非 MatMul 耗时 Top Kernels (列举 2-3 个):
#       1. at::native::elementwise_kernel / vectorized_elementwise_kernel [对应 RoPE, Activation, Residual Add]
#       2. at::native::reduce_kernel [对应 RMSNorm, Softmax]
#       3. at::native::index_elementwise_kernel [对应 Embedding/RoPE 索引]

# (d) 训练步 vs. 推理步 (Training vs. Inference)
# 问题: 对比完整的训练步（Forward + Backward + Optimizer）和仅推理（Forward Pass）。
#       矩阵乘法所占的时间比例发生了怎样的变化？其他 kernels 呢？
#
# [填空] Forward Pass 中 MatMul 占比 (约 %): 较高 (Linear 占主导)
# [填空] Full Training Step 中 MatMul 占比 (约 %): 略微下降
# [分析] 变化趋势 (增加/减少/不变): 略微下降
# [分析] 其他 kernels (如 element-wise, norm) 的占比变化: Optimizer Step (4.2% Time) 引入了大量的 Element-wise 操作，稀释了 MatMul 的占比。

# (e) Softmax vs. Matrix Multiplication (in Self-Attention)
# 问题: 比较 Forward Pass 期间，Self-Attention 层内部 Softmax 操作与矩阵乘法操作的运行时间。
#       运行时间的差异与 FLOPs 的差异相比如何？
#
# [填空] Self-Attention 中 Softmax 总耗时: 22.0 ms (Total)
# [填空] Self-Attention 中 MatMul (Q*K + Attn*V) 总耗时: 32.9 + 11.2 = 44.1 ms (Total)
# [分析] 耗时差异 (MatMul 是 Softmax 的多少倍?): 约 2 倍
# [分析] FLOPs 差异 (理论上 MatMul FLOPs 远大于 Softmax): 理论差异巨大 (>64倍)
#       虽然 FLOPs 差异巨大，但运行时间差异是否没那么大？这是因为 Softmax 是 Memory-bound (带宽受限)？


