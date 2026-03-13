# 文件功能：
#   核心梯度计算引擎。
#   对 Expert A / Expert B 分别执行完整的前向+反向传播，
#   逐 Batch 累加真实梯度敏感度矩阵 S = |grad * weight|，
#   最终将 S_A、S_B 序列化为 .pt 文件供后续合并使用。
#
# 需要实现的函数：
#
# load_model_with_grad(
#     model_name_or_path: str,
#     accelerator           # accelerate.Accelerator 实例
# ) -> tuple[nn.Module, PreTrainedTokenizer]
#   输入：model_name_or_path — HuggingFace 模型路径或 Hub ID
#         accelerator        — 已初始化的 Accelerator（负责多卡分片）
#   输出：(model, tokenizer)
#         model 的 Attention/MLP 层 requires_grad=True，其余层 False
#
# compute_sensitivity(
#     model: nn.Module,
#     dataloader: DataLoader,
#     accelerator,
#     use_fisher: bool = False
# ) -> dict[str, Tensor]
#   输入：model      — load_model_with_grad 返回的模型
#         dataloader — data_loader.build_dataloader 返回的 DataLoader
#         accelerator
#         use_fisher — True 时用 Fisher 近似 S=G²·|W|，False 时用 S=|G·W|
#   输出：sensitivity_dict，key 为 param_name(str)，value 为与 param 同形的 Tensor（CPU）
#         即 S_total 累加完毕后 detach().cpu() 的结果
#
# save_sensitivity(
#     sensitivity_dict: dict[str, Tensor],
#     save_path: str
# ) -> None
#   输入：sensitivity_dict — compute_sensitivity 的输出
#         save_path        — 目标 .pt 文件路径（如 "outputs/S_A.pt"）
#   输出：无（副作用：写文件）
