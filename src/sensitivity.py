"""
功能：基于 hooks.py 收集的激活统计量，计算每层参数的近似敏感度矩阵 S_approx，
      并计算两个专家模型之间的逐层冲突率（Conflict Rate），供 gls_router.py 决策使用。

公式：
    S_approx[layer] = |W[layer]| * ||X[layer]||_2        (element-wise)
    Conflict(A, B)  = 1 - |S_A - S_B| / (S_A + S_B)     (layer-level scalar)

需要实现的函数：

def compute_sensitivity(
    model,
    activation_stats: dict[str, "torch.Tensor"]
) -> dict[str, "torch.Tensor"]:
    """
    计算模型每层的近似敏感度矩阵。

    输入：
        model            : nn.Module          — 专家模型（A 或 B）
        activation_stats : dict[str, Tensor]  — 来自 hooks.ActivationHookManager.get_activation_stats()
                                                key = 层名称，value = Tensor[in_features]（激活 L2 均值）

    输出：
        dict[str, Tensor] — key = 层名称，value = Tensor[out_features, in_features]
                            即 S_approx 矩阵，与对应层权重 W 形状相同
    """
    pass


def compute_layer_sensitivity_scalar(
    sensitivity: dict[str, "torch.Tensor"]
) -> dict[str, float]:
    """
    将逐元素敏感度矩阵聚合为每层的标量，用于 Conflict Rate 计算。

    输入：
        sensitivity : dict[str, Tensor] — 来自 compute_sensitivity() 的输出

    输出：
        dict[str, float] — key = 层名称，value = 该层所有元素敏感度之和（标量）
    """
    pass


def compute_conflict_rate(
    scalar_A: dict[str, float],
    scalar_B: dict[str, float]
) -> dict[str, float]:
    """
    计算两个专家模型在每层的冲突率。

    输入：
        scalar_A : dict[str, float] — 来自 compute_layer_sensitivity_scalar()，专家 A 的层级标量
        scalar_B : dict[str, float] — 来自 compute_layer_sensitivity_scalar()，专家 B 的层级标量

    输出：
        dict[str, float] — key = 层名称，value = Conflict ∈ [0, 1]，越接近 1 冲突越大
    """
    pass
"""
