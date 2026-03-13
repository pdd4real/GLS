"""
功能：第一阶段主脚本。
      加载专家模型 A 和 B，在校准数据集上执行纯前向推理（no_grad），
      通过 hooks.ActivationHookManager 收集激活统计量，
      调用 sensitivity.compute_sensitivity() 计算 S_approx，
      最终将 S_A、S_B 及 conflict_rates 序列化保存为 .pt 文件，供第二阶段使用。

运行方式：
    python scripts/run_activation_cache.py \
        --model_a  <path_or_hub_id_A> \
        --model_b  <path_or_hub_id_B> \
        --calib    data/calib_mixed_150.jsonl \
        --out_dir  data/

命令行参数：
    --model_a  : str — 专家 A 的模型路径或 HuggingFace Hub ID
    --model_b  : str — 专家 B 的模型路径或 HuggingFace Hub ID
    --calib    : str — 校准数据 JSONL 路径，默认 "data/calib_mixed_150.jsonl"
    --out_dir  : str — 输出目录，默认 "data/"
    --batch_size : int — 推理 batch size，默认 4

输出文件（保存到 out_dir）：
    sensitivity_A.pt    — dict[str, Tensor]，来自 sensitivity.compute_sensitivity()
    sensitivity_B.pt    — dict[str, Tensor]
    conflict_rates.pt   — dict[str, float]，来自 sensitivity.compute_conflict_rate()

调用链：
    utils.model_io.load_model()
    -> src.data_loader.CalibDataLoader
    -> src.hooks.ActivationHookManager
    -> src.sensitivity.compute_sensitivity()
    -> src.sensitivity.compute_layer_sensitivity_scalar()
    -> src.sensitivity.compute_conflict_rate()
    -> utils.model_io.save_sensitivity()
"""
