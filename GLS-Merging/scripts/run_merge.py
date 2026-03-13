# 文件功能：
#   CPU/单卡轻量脚本：读取预计算好的 S_A.pt / S_B.pt 和两个模型的 state_dict，
#   执行 Coarse-to-Fine 权重合并，保存合并后的模型。
#   无需 GPU，秒级完成，可反复调整阈值参数。
#
# 执行方式：
#   python scripts/run_merge.py \
#       --model_a RLHFlow/LLaMA3-iterative-DPO-final \
#       --model_b TechxGenus/Meta-Llama-3-8B-Instruct-Coder \
#       --sens_a  outputs/S_A.pt \
#       --sens_b  outputs/S_B.pt \
#       --conflict_threshold 0.8 \
#       --output_dir outputs/merged_model/
#
# 主要步骤（在 main() 中实现）：
#   1. 调用 model_io.load_state_dict_cpu 加载 state_dict_A / state_dict_B
#   2. torch.load S_A.pt → sensitivity_dict_A，S_B.pt → sensitivity_dict_B
#   3. 调用 sensitivity.aggregate_layer_sensitivity 得到 layer_sens_A / layer_sens_B
#   4. 调用 sensitivity.compute_conflict_rate → conflict_dict
#   5. 调用 sensitivity.classify_layers(conflict_dict, threshold) → routing_plan
#   6. 调用 gls_router.merge_models → merged_state_dict
#   7. 调用 model_io.save_merged_model 写出合并模型
