# 文件功能：
#   GLS 合并总调度器（Coarse-to-Fine Router）。
#   读取 routing_plan，对每一层调用对应的合并方法，
#   将合并后的权重写回新模型的 state_dict 并返回。
#
# 需要实现的函数：
#
# merge_models(
#     state_dict_A: dict[str, Tensor],
#     state_dict_B: dict[str, Tensor],
#     sensitivity_dict_A: dict[str, Tensor],
#     sensitivity_dict_B: dict[str, Tensor],
#     routing_plan: dict[str, str]
# ) -> dict[str, Tensor]
#   输入：state_dict_A/B       — Expert A/B 的完整 state_dict（CPU Tensor）
#         sensitivity_dict_A/B — gradient_engine.compute_sensitivity 的输出（CPU Tensor）
#         routing_plan         — sensitivity.classify_layers 的输出
#                                key 为层名前缀，value 为 "layer_wise" | "channel_wise"
#   输出：merged_state_dict，结构与 state_dict_A 相同，权重已按路由策略合并
#         非 Attention/MLP 参数（如 embed_tokens、norm）直接从 state_dict_A 复制
