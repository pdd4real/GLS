"""
功能：细粒度（Channel-wise / Head-wise）合并逻辑。
      当某层 Conflict Rate >= 阈值时，对该层按输出通道（列）分别计算混合系数，
      实现更精细的参数级别合并，避免高冲突层的能力互相干扰。

合并公式（逐通道）：
    alpha_A[c] = S_A[c] / (S_A[c] + S_B[c])      for each output channel c
    W_new[c, :] = alpha_A[c] * W_A[c, :] + (1 - alpha_A[c]) * W_B[c, :]

需要实现的函数：

def channel_wise_merge(
    W_A: "torch.Tensor",
    W_B: "torch.Tensor",
    S_A: "torch.Tensor",
    S_B: "torch.Tensor"
) -> "torch.Tensor":
    """
    对单层权重执行 channel-wise 加权合并。

    输入：
        W_A : Tensor[out, in] — 专家 A 该层权重矩阵
        W_B : Tensor[out, in] — 专家 B 该层权重矩阵（形状与 W_A 相同）
        S_A : Tensor[out, in] — 专家 A 该层敏感度矩阵，来自 sensitivity.compute_sensitivity()
        S_B : Tensor[out, in] — 专家 B 该层敏感度矩阵（形状与 S_A 相同）

    输出：
        Tensor[out, in] — 合并后的权重矩阵，形状与输入相同
    """
    pass
"""
