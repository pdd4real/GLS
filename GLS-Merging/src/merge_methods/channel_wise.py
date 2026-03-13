# 文件功能：
#   对冲突率高的层执行 Channel-wise（Output Channel 维度）权重合并。
#   按列切片敏感度矩阵，对每个 output channel 独立计算混合系数后拼接。
#
# 需要实现的函数：
#
# channel_wise_merge(
#     W_A: Tensor,
#     W_B: Tensor,
#     S_A: Tensor,
#     S_B: Tensor
# ) -> Tensor
#   输入：W_A, W_B — 同形状权重矩阵 (out_features, in_features)
#         S_A, S_B — 与 W_A/W_B 同形状的敏感度矩阵（来自 gradient_engine，未聚合的原始矩阵）
#   输出：W_merged — 与输入同形状的合并权重 Tensor
#         每行（output channel）独立按 α_i = S_A[i].sum() / (S_A[i].sum() + S_B[i].sum()) 加权
