"""
功能：第二阶段主脚本。
      从磁盘加载预计算好的 S_A、S_B、conflict_rates（.pt 文件），
      调用 gls_router.route_and_merge() 执行 Coarse-to-Fine 动态合并，
      将合并后的模型保存到指定目录。

运行方式：
    python scripts/run_merge.py \
        --model_a   <path_or_hub_id_A> \
        --model_b   <path_or_hub_id_B> \
        --sens_a    data/sensitivity_A.pt \
        --sens_b    data/sensitivity_B.pt \
        --conflicts data/conflict_rates.pt \
        --out_dir   outputs/merged_model \
        --threshold 0.8

命令行参数：
    --model_a   : str   — 专家 A 的模型路径或 HuggingFace Hub ID
    --model_b   : str   — 专家 B 的模型路径或 HuggingFace Hub ID
    --sens_a    : str   — sensitivity_A.pt 路径，默认 "data/sensitivity_A.pt"
    --sens_b    : str   — sensitivity_B.pt 路径，默认 "data/sensitivity_B.pt"
    --conflicts : str   — conflict_rates.pt 路径，默认 "data/conflict_rates.pt"
    --out_dir   : str   — 合并模型保存目录，默认 "outputs/merged_model"
    --threshold : float — Conflict Rate 阈值，默认 0.8

调用链：
    utils.model_io.load_model()
    -> utils.model_io.load_sensitivity()
    -> src.merge_methods.gls_router.route_and_merge()
    -> utils.model_io.save_model()
"""
