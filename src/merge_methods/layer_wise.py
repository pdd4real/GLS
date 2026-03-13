"""
功能：粗粒度（Layer-wise）合并逻辑。
      当某层 Conflict Rate < 阈值时，对该层所有参数使用统一的混合系数 alpha 进行加权合并。

合并公式：
    alpha_A = S_layer_A / (S_layer_A + S_layer_B)
    W_new   = alpha_A * W_A + (1 - alpha_A) * W_B

需要实现的函数：

def layer_wise_merge(
    W_A: "torch.Tensor",
    W_B: "torch.Tensor",
    s_scalar_A: float,
    s_scalar_B: float
) -> "torch.Tensor":
    """
    对单层权重执行 layer-wise 加权合并。

    输入：
        W_A        : Tensor[out, in] — 专家 A 该层权重矩阵
        W_B        : Tensor[out, in] — 专家 B 该层权重矩阵（形状与 W_A 相同）
        s_scalar_A : float           — 专家 A 该层敏感度标量，来自 sensitivity.compute_layer_sensitivity_scalar()
        s_scalar_B : float           — 专家 B 该层敏感度标量

    输出：
        Tensor[out, in] — 合并后的权重矩阵，形状与输入相同
    """
    pass
"""
