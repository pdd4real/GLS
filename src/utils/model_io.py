"""
功能：封装 HuggingFace 模型的高效加载与保存操作，统一管理模型/tokenizer 的 I/O，
      供 run_activation_cache.py 和 run_merge.py 调用。

需要实现的函数：

def load_model(
    model_name_or_path: str,
    dtype: str = "float16",
    device_map: str = "auto"
) -> tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    """
    加载 HuggingFace 模型与对应 tokenizer。

    输入：
        model_name_or_path : str — HuggingFace Hub ID 或本地路径
        dtype              : str — 加载精度，"float16" / "bfloat16" / "float32"
        device_map         : str — 设备映射策略，"auto" 自动分配到可用 GPU

    输出：
        (model, tokenizer) : tuple
            model     : PreTrainedModel       — 已加载到设备的模型，eval 模式
            tokenizer : PreTrainedTokenizer   — 对应 tokenizer，pad_token 已设置
    """
    pass


def save_model(
    model,
    tokenizer,
    save_path: str
) -> None:
    """
    将合并后的模型与 tokenizer 保存到本地目录（HuggingFace safetensors 格式）。

    输入：
        model     : PreTrainedModel     — 合并后的模型
        tokenizer : PreTrainedTokenizer — 对应 tokenizer
        save_path : str                 — 保存目录路径

    输出：
        None（副作用：写入磁盘）
    """
    pass


def save_sensitivity(
    sensitivity: dict[str, "torch.Tensor"],
    save_path: str
) -> None:
    """
    将敏感度字典序列化为 .pt 文件，避免重复前向推理。

    输入：
        sensitivity : dict[str, Tensor] — 来自 sensitivity.compute_sensitivity()
        save_path   : str               — 目标 .pt 文件路径，如 "data/sensitivity_A.pt"

    输出：
        None（副作用：写入磁盘）
    """
    pass


def load_sensitivity(
    load_path: str
) -> dict[str, "torch.Tensor"]:
    """
    从 .pt 文件加载敏感度字典。

    输入：
        load_path : str — .pt 文件路径

    输出：
        dict[str, Tensor] — 与 sensitivity.compute_sensitivity() 输出格式相同
    """
    pass
"""
