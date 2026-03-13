# 文件功能：
#   一次性脚本：从 openai_humaneval / gsm8k / alpaca 各采样若干条，
#   格式化为 Llama-3 Chat Template，写入 data/calib_mixed_150.jsonl。
#
# 执行方式：python scripts/prepare_calib_data.py --output data/calib_mixed_150.jsonl
#
# 主要步骤（在 main() 中实现）：
#   1. 用 datasets.load_dataset 加载三个数据集，各随机抽取 50 条
#   2. 统一提取 instruction 字段，标注 domain（"code"/"math"/"general"）
#   3. 调用 data_loader.format_to_chat_template 格式化
#   4. 写入 JSONL，每行格式：{"text": str, "domain": str}
