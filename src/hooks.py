"""
功能：通过 PyTorch register_forward_hook 机制，在模型前向推理时拦截并累积
      各线性层的输入激活值（Activation），供后续计算近似敏感度 S_approx 使用。

需要实现的类/函数：

class ActivationHookManager:
    """
    管理所有层的 hook 注册、激活值累积与清理。
    """

    def __init__(self, model, target_module_types: tuple = None):
        """
        输入：
            model               : nn.Module — 已加载的 HuggingFace 模型
            target_module_types : tuple     — 需要挂载 hook 的层类型，默认为 (nn.Linear,)
                                             对应 q_proj / v_proj / mlp.down_proj 等
        属性（初始化后可访问）：
            self.activation_stats : dict[str, Tensor]
                key   = 层名称（module full name）
                value = Tensor[out_features] — 累积的激活 L2 范数均值（跨所有 batch）
        """
        pass

    def register(self):
        """
        遍历 model 的所有目标层，挂载 forward hook。
        无输入/输出，副作用：填充 self._hooks 列表。
        """
        pass

    def remove(self):
        """
        移除所有已注册的 hook，释放资源。
        无输入/输出。
        """
        pass

    def get_activation_stats(self) -> dict[str, "torch.Tensor"]:
        """
        返回累积完毕的激活统计量。

        输出：
            dict[str, Tensor] — key = 层名称，value = Tensor[in_features]
                                表示该层输入激活的平均 L2 范数（已对 batch/seq 维度取均值）
        """
        pass


def _make_hook(name: str, stats_store: dict) -> callable:
    """
    工厂函数，生成单个层的 forward hook 闭包。

    输入：
        name        : str  — 层名称，用作 stats_store 的 key
        stats_store : dict — 共享的激活统计存储（即 ActivationHookManager.activation_stats）

    输出：
        callable — 符合 PyTorch hook 签名 hook(module, input, output) 的函数
                   该函数计算 input[0] 在 (batch, seq) 维度上的 L2 范数并累加到 stats_store[name]
    """
    pass
"""
