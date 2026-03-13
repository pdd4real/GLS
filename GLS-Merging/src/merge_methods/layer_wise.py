# 文件功能：
#   对冲突率低的层执行 Layer-wise 权重合并。
#   合并策略：按敏感度归一化加权平均 W_merged = α·W_A + (1-α)·W_B，
#   其中 α = S_A / (S_A + S_B)。
#
# 需要实现的函数：
#
# layer_wise_merge(
#     W_A: Tensor,
#     W_B: Tensor,
#     s_A: float,
#     s_B: float
# ) -> Tensor
#   输入：W_A, W_B — 同形状的权重矩阵 (out_features, in_features)
#         s_A, s_B — 对应层的标量敏感度（来自 sensitivity.aggregate_layer_sensitivity）
#   输出：W_merged — 与 W_A/W_B 同形状的合并权重 Tensor
