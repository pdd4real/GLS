# 文件功能：
#   读取 gradient_engine 输出的敏感度字典，
#   计算每一层的 Layer-wise Conflict Rate，
#   决定该层应走 layer_wise 还是 channel_wise 合并路径。
#
# 需要实现的函数：
#
# aggregate_layer_sensitivity(
#     sensitivity_dict: dict[str, Tensor]
# ) -> dict[str, float]
#   输入：sensitivity_dict — gradient_engine.compute_sensitivity 的输出
#         key 格式示例："model.layers.0.self_attn.q_proj.weight"
#   输出：layer_sens，key 为层名前缀（如 "model.layers.0"），value 为该层所有参数 S 之和（float）
#
# compute_conflict_rate(
#     layer_sens_A: dict[str, float],
#     layer_sens_B: dict[str, float]
# ) -> dict[str, float]
#   输入：layer_sens_A / layer_sens_B — aggregate_layer_sensitivity 对 Expert A/B 的输出
#   输出：conflict_dict，key 为层名，value 为 Conflict Rate ∈ [0,1]
#         公式：1 - |S_A - S_B| / (S_A + S_B)
#
# classify_layers(
#     conflict_dict: dict[str, float],
#     threshold: float = 0.8
# ) -> dict[str, str]
#   输入：conflict_dict — compute_conflict_rate 的输出
#         threshold     — 高冲突判定阈值
#   输出：routing_plan，key 为层名，value 为 "layer_wise" 或 "channel_wise"
