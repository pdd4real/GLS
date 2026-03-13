"""
功能：Coarse-to-Fine 动态路由器。
      读取每层的 Conflict Rate，根据阈值决定调用 layer_wise 还是 channel_wise 合并，
      最终输出完整的合并后模型权重字典。

预留接口：
    - 正交子空间投影（Orthogonal Subspace Projection）后处理钩子
    - 带温度的 Softmax 正则化 alpha 计算（替换默认线性归一化）

需要实现的函数：

def route_and_merge(
    model_A,
    model_B,
    sensitivity_A: dict[str, "torch.Tensor"],
    sensitivity_B: dict[str, "torch.Tensor"],
    conflict_rates: dict[str, float],
    conflict_threshold: float = 0.8,
    postprocess_fn: callable = None
) -> dict[str, "torch.Tensor"]:
    """
    遍历所有层，根据冲突率动态选择合并粒度，返回合并后的 state_dict。

    输入：
        model_A             : nn.Module          — 专家 A 模型
        model_B             : nn.Module          — 专家 B 模型
        sensitivity_A       : dict[str, Tensor]  — 来自 sensitivity.compute_sensitivity()，专家 A
        sensitivity_B       : dict[str, Tensor]  — 来自 sensitivity.compute_sensitivity()，专家 B
        conflict_rates      : dict[str, float]   — 来自 sensitivity.compute_conflict_rate()
        conflict_threshold  : float              — 高/低冲突分界阈值，默认 0.8
        postprocess_fn      : callable | None    — 可选后处理函数，签名为
                                                   fn(layer_name, W_merged, W_A, W_B) -> Tensor
                                                   用于接入正交投影或 Softmax 正则化（消融实验预留）

    输出：
        dict[str, Tensor] — 合并后的完整 state_dict，可直接传入 model.load_state_dict()
    """
    pass
"""
