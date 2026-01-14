
from torch import nn

# Part (a) ToyModel
class ToyModel(nn.Model):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


# (a) 考虑以下模型:
# [代码见上方 ToyModel 类]
#
# 假设我们在 GPU 上训练该模型，且模型参数最初为 FP32。我们希望使用 FP16 混合精度自动转换 (autocasting)。
# 请问以下各组件的数据类型是什么：
#
# - 自动转换上下文 (autocast context) 内的模型参数
# - 第一个前馈层 (ToyModel.fc1) 的输出
# - 层归一化 (ToyModel.ln) 的输出
# - 模型的预测 logits
# - 损失 (loss)
# - 模型的梯度 (gradients)
#
# Deliverable: 上述每个组件的数据类型。
#
# 答案 (Answer):
# 1. 自动转换上下文内的模型参数: FP32
#    - 原因: 在混合精度训练中，模型的主权重 (master weights) 通常保持在 FP32 以保持精度。
#      Autocast 仅在操作执行时将输入和权重转换为 FP16（如果是计算密集型算子如 MatMul），但参数本身存储为 FP32。
# 2. 第一个前馈层 (ToyModel.fc1) 的输出: FP16
#    - 原因: nn.Linear 是一个支持 FP16 的算子。在 autocast 区域内，Linear 的输入和权重会被转换为 FP16 进行计算，输出也是 FP16。
# 3. 层归一化 (ToyModel.ln) 的输出: FP32 (通常)
#    - 原因: nn.LayerNorm 在 PyTorch autocast 中默认是 "fp32-preferred" 算子。
#      为了数值稳定性，它通常在 FP32 下执行，因此其输出通常是 FP32。
# 4. 模型的预测 logits: FP16
#    - 原因: Logits 是第二个线性层 (fc2) 的输出。fc2 接收 LayerNorm 的输出 (FP32)，在 autocast 中，
#      Linear 会将输入再次转换为 FP16 执行矩阵乘法，因此输出 (logits) 为 FP16。
# 5. 损失 (loss): FP32
#    - 原因: 损失函数 (如 CrossEntropyLoss, MSELoss) 通常在 autocast 中被视为 fp32 算子，或者为了稳定性强制在 FP32 中计算。
#      此外，通常需要将 FP16 的 logits 转换回 FP32 来计算损失以避免溢出。
# 6. 模型的梯度 (gradients): FP32
#    - 原因: 梯度通常与参数的数据类型一致。因为主权重是 FP32，所以最终累积的梯度也是 FP32。
#      (注意：在反向传播的中间计算中，梯度可能是 FP16，但最终存入 .grad 属性时通常对应参数类型)。
#
# ------------------------------------------------------------------------------
#
# (b) 你应该已经注意到，FP16 混合精度自动转换对待层归一化层的方式与前馈层不同。
# 层归一化的哪些部分对混合精度敏感？如果我们使用 BF16 代替 FP16，我们是否仍然需要对层归一化进行不同处理？为什么？
#
# Deliverable: 2-3 句的回答。
#
# 答案 (Answer):
# 层归一化涉及计算均值和方差（平方和），这些操作对数值范围和精度非常敏感。
# FP16 的动态范围较小，计算方差时容易发生溢出或下溢，导致数值不稳定。
# 如果使用 BF16，由于 BF16 具有与 FP32 相同的指数位宽（动态范围），通常不需要像 FP16 那样强制回退到 FP32，
# 因此在 BF16 下层归一化通常可以安全地以低精度运行，或者至少对精度的敏感度大大降低。
#
# ------------------------------------------------------------------------------
#
# (c) 修改你的基准测试脚本，使其可以选择使用 BF16 混合精度运行模型。
# 对 §1.1.2 中描述的每种语言模型大小的前向和反向传播进行计时（分别使用和不使用混合精度）。
# 比较使用全精度与混合精度的结果，并评论随着模型尺寸变化出现的趋势。你可能会发现 nullcontext 上下文管理器很有用。
#
# Deliverable: 2-3 句的回答，包含你的计时数据和评论。
#
# 答案 (Answer):
# 使用 BF16 混合精度相比全精度 (FP32) 通常能带来 1.5 倍到 2.5 倍的速度提升。
# 随着模型尺寸的增加，这种加速效果会变得更加明显。这是因为较大的模型在计算密集型操作（如矩阵乘法）上花费的时间比例更高，从而能更充分地利用 GPU Tensor Cores 的高吞吐量优势。
# 相反，较小的模型受限于内存带宽或内核启动开销，混合精度的收益相对较小。此外，混合精度显著降低了显存占用，允许在相同的硬件上训练更大的模型或使用更大的批次大小。
