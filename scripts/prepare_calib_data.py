"""
功能：从 HuggingFace 下载校准数据集（HumanEval / GSM8K / Alpaca），
      各抽取 50 条，统一格式化为 Llama-3 Chat Template，
      保存为 data/calib_mixed_150.jsonl。

运行方式：
    python scripts/prepare_calib_data.py

依赖：
    - datasets（HuggingFace）
    - transformers（用于 tokenizer.apply_chat_template）
    - src/data_loader.apply_chat_template

输出文件：
    data/calib_mixed_150.jsonl
        每行一个 JSON 对象，字段：
            {
                "text"   : str,   # 已格式化的 Chat Template 字符串
                "domain" : str    # "code" | "math" | "general"
            }
"""
