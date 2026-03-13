# 文件功能：
#   GPU 密集型脚本：对 Expert A 和 Expert B 分别执行完整前向+反向传播，
#   提取真实梯度敏感度 S，保存为 outputs/S_A.pt 和 outputs/S_B.pt。
#   此脚本需在多卡环境下用 accelerate launch 执行。
#
# 执行方式：
#   accelerate launch scripts/run_gradient_cache.py \
#       --model_a RLHFlow/LLaMA3-iterative-DPO-final \
#       --model_b TechxGenus/Meta-Llama-3-8B-Instruct-Coder \
#       --calib_data data/calib_mixed_150.jsonl \
#       --output_dir outputs/
#
# 主要步骤（在 main() 中实现）：
#   1. 初始化 Accelerator
#   2. 调用 gradient_engine.load_model_with_grad 加载 Expert A，
#      调用 data_loader.build_dataloader 构建 DataLoader
#   3. 调用 gradient_engine.compute_sensitivity 得到 S_A
#   4. 调用 gradient_engine.save_sensitivity(S_A, "outputs/S_A.pt")
#   5. 对 Expert B 重复步骤 2-4，得到 S_B
