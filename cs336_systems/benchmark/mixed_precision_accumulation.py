import torch

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)

s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)

"""
结果分析：
纯 FP16 累加由于精度不足产生了显著的累积误差（结果为 9.9531），而纯 FP32 累加结果（10.0001）非常接近 10。
混合精度累加（将 FP16 转换后累加到 FP32）虽然受限于 FP16 对 0.01 的初始表示误差，但避免了累加过程中的精度损失，结果（10.0021）远比纯 FP16 准确。
"""
