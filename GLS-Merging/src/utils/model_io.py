# 文件功能：
#   模型权重的 I/O 工具，封装 HuggingFace + Accelerate/DeepSpeed 的加载与保存细节，
#   使上层代码无需关心分布式权重分片。
#
# 需要实现的函数：
#
# load_state_dict_cpu(
#     model_name_or_path: str
# ) -> dict[str, Tensor]
#   输入：model_name_or_path — HuggingFace 模型路径或 Hub ID
#   输出：完整 state_dict，所有 Tensor 在 CPU 上（用于 run_merge.py 的纯 CPU 合并）
#
# save_merged_model(
#     merged_state_dict: dict[str, Tensor],
#     base_model_name_or_path: str,
#     save_dir: str
# ) -> None
#   输入：merged_state_dict      — gls_router.merge_models 的输出
#         base_model_name_or_path — 用于复制 config/tokenizer 的基础模型路径
#         save_dir               — 输出目录
#   输出：无（副作用：将合并模型以 HuggingFace 格式写入 save_dir）
