"""
(a) 编写一个脚本，对模型的前向和反向传播进行基本的端到端基准测试。具体来说，您的脚本应支持以下功能：
    - 给定超参数（例如，层数），初始化一个模型。
    - 生成一批随机数据。
    - 运行 w 个预热步骤（在开始计时之前），然后计时 n 个步骤的执行（仅前向传播，或前向和反向传播，具体取决于参数）。
      为了计时，可以使用 Python 的 timeit 模块（例如，使用 timeit 函数，或使用 timeit.default_timer()，
      它提供系统最高分辨率的时钟，因此是比 time.time() 更好的基准测试默认值）。
    - 在每一步之后调用 torch.cuda.synchronize()。

Deliverable: 一个脚本，它将初始化一个具有给定超参数的 basics Transformer 模型，创建一批随机数据，并对前向和反向传播进行计时。
"""

from cs336_basics.model import BasicsTransformerLM  # pyright: ignore[reportMissingImports]
import argparse
import torch
from timeit import default_timer as timer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--d_model", type=int, default=768)
parser.add_argument("--d_ff", type=int, default=3072)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--num_heads", type=int, default=12)
parser.add_argument("--warmup_iterations", type=int, default=10)
parser.add_argument("--benchmark_iterations", type=int, default=100)
parser.add_argument("--forward_or_backward", type=str, default="forward", choices=["forward", "backward", "both"])
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--context_length", type=int, default=1024)
parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])

args = parser.parse_args()

if args.dtype == "bfloat16":
    dtype = torch.bfloat16
elif args.dtype == "float16":
    dtype = torch.float16
else:
    dtype = torch.float32

print(f"Initializing model with dtype={dtype}")

model = BasicsTransformerLM(
    d_model=args.d_model,
    d_ff=args.d_ff,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    vocab_size=50257,  # Default GPT-2 vocab size
    context_length=args.context_length,
    rope_theta=10000.0,
).to("cuda", dtype=dtype)

# random data
test_data = torch.randint(0, 50257, (args.batch_size, args.context_length), device="cuda")

# warm-up
warmup_iterations = args.warmup_iterations
model.eval()
print("Starting warmup...")
start = timer()
with torch.no_grad():
    # For warmup, we also use mixed precision context if needed,
    # but since model is already cast to dtype, inputs (int) are fine.
    # We might need autocast for some ops to be happy, but usually .to(dtype) is enough for weights.
    for _ in range(warmup_iterations):
        model(test_data)
        torch.cuda.synchronize()
warmup_time = timer() - start
print(f"Warmup time: {warmup_time:.2f} seconds")

# benchmark
benchmark_iterations = args.benchmark_iterations
forward_or_backward = args.forward_or_backward


def calculate_stats(times):
    if not times:
        return 0.0, 0.0
    # Use numpy for stable std calculation
    mean = np.mean(times)
    std = np.std(times)
    return mean, std


def run_forward():
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(benchmark_iterations):
            start = timer()
            model(test_data)
            torch.cuda.synchronize()
            end = timer()
            times.append(end - start)
    return calculate_stats(times)


def run_backward():
    model.train()
    forward_times = []
    backward_times = []
    for _ in range(benchmark_iterations):
        model.zero_grad()

        start_fwd = timer()
        output = model(test_data)
        torch.cuda.synchronize()
        end_fwd = timer()
        forward_times.append(end_fwd - start_fwd)

        loss = output.sum()

        start_bwd = timer()
        loss.backward()
        torch.cuda.synchronize()
        end_bwd = timer()
        backward_times.append(end_bwd - start_bwd)

    return calculate_stats(forward_times), calculate_stats(backward_times)


if forward_or_backward == "forward":
    mean, std = run_forward()
    print(f"Forward time: mean={mean:.4f} s, std={std:.4f} s")
elif forward_or_backward == "backward":
    (f_mean, f_std), (b_mean, b_std) = run_backward()
    print(f"Forward pass time (in training): mean={f_mean:.4f} s, std={f_std:.4f} s")
    print(f"Backward pass time: mean={b_mean:.4f} s, std={b_std:.4f} s")
elif forward_or_backward == "both":
    f_mean_eval, f_std_eval = run_forward()
    print(f"Forward time (eval): mean={f_mean_eval:.4f} s, std={f_std_eval:.4f} s")

    # Clear cache before backward to free up memory from forward benchmark
    torch.cuda.empty_cache()

    (f_mean_train, f_std_train), (b_mean, b_std) = run_backward()
    print(f"Forward time (train): mean={f_mean_train:.4f} s, std={f_std_train:.4f} s")
    print(f"Backward time: mean={b_mean:.4f} s, std={b_std:.4f} s")
