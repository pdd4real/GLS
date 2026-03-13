"""
功能：加载本地校准数据集，并将其 tokenize 为模型可接受的输入格式。

需要实现的类/函数：

class CalibDataLoader:
    """
    封装校准数据的加载与 tokenize 逻辑。
    """

    def __init__(self, data_path: str, tokenizer, batch_size: int = 8, max_length: int = 512):
        """
        输入：
            data_path   : str  — 本地 JSONL 文件路径，如 "data/calib_mixed_150.jsonl"
            tokenizer   : PreTrainedTokenizer — HuggingFace tokenizer（已加载）
            batch_size  : int  — 每批样本数，默认 8
            max_length  : int  — 最大 token 长度，默认 512
        """
        pass

    def __iter__(self):
        """
        迭代器，每次 yield 一个 batch 的 tokenized 输入。

        输出（每次 yield）：
            batch : dict — {"input_ids": Tensor[B, L], "attention_mask": Tensor[B, L]}
                          B = batch_size，L = max_length（padding/truncation 后）
        """
        pass

    def __len__(self) -> int:
        """
        输出：
            int — 总 batch 数
        """
        pass


def load_jsonl(path: str) -> list[dict]:
    """
    读取 JSONL 文件，每行解析为一个 dict。

    输入：
        path : str — JSONL 文件路径

    输出：
        list[dict] — 每条样本，字段至少包含 "text" 或 "prompt"
    """
    pass


def apply_chat_template(sample: dict, tokenizer) -> str:
    """
    将单条原始样本转换为 Llama-3 Chat Template 格式的字符串。

    输入：
        sample    : dict — 原始样本，字段视来源数据集而定（含 "instruction"/"prompt" 等）
        tokenizer : PreTrainedTokenizer — 用于调用 apply_chat_template

    输出：
        str — 格式化后的完整对话字符串
    """
    pass
"""
