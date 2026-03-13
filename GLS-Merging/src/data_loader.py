# 文件功能：
#   加载并预处理校准数据集 calib_mixed_150.jsonl，
#   将原始 JSONL 格式化为 Llama-3 Chat Template，
#   返回可供 gradient_engine.py 直接消费的 DataLoader。
#
# 需要实现的函数：
#
# load_calib_dataset(path: str) -> list[dict]
#   输入：path — calib_mixed_150.jsonl 的文件路径
#   输出：原始样本列表，每条为 {"instruction": str, "domain": str}
#
# format_to_chat_template(samples: list[dict], tokenizer) -> list[str]
#   输入：samples — load_calib_dataset 的输出
#         tokenizer — HuggingFace tokenizer（含 apply_chat_template 方法）
#   输出：已格式化的字符串列表（Llama-3 Chat Template 格式）
#
# build_dataloader(
#     path: str,
#     tokenizer,
#     batch_size: int = 4,
#     max_length: int = 512
# ) -> torch.utils.data.DataLoader
#   输入：path       — JSONL 文件路径
#         tokenizer  — HuggingFace tokenizer
#         batch_size — 每批样本数
#         max_length — token 截断长度
#   输出：DataLoader，每个 batch 为 dict{"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor}
#         labels 与 input_ids 相同（Causal LM next-token prediction）
