# GLS-Merging

Gradient-based Layer-Sensitive model merging with Coarse-to-Fine dynamic granularity.

## Pipeline

```
prepare_calib_data.py          # 一次性：采样并格式化校准数据
        ↓
run_gradient_cache.py          # GPU密集：计算真实梯度敏感度 → S_A.pt / S_B.pt
        ↓
run_merge.py                   # CPU秒级：读取 .pt，Coarse-to-Fine 合并 → merged_model/
```

## Quick Start

```bash
pip install -r requirements.txt

python scripts/prepare_calib_data.py --output data/calib_mixed_150.jsonl

accelerate launch scripts/run_gradient_cache.py \
    --model_a RLHFlow/LLaMA3-iterative-DPO-final \
    --model_b TechxGenus/Meta-Llama-3-8B-Instruct-Coder \
    --calib_data data/calib_mixed_150.jsonl \
    --output_dir outputs/

python scripts/run_merge.py \
    --model_a RLHFlow/LLaMA3-iterative-DPO-final \
    --model_b TechxGenus/Meta-Llama-3-8B-Instruct-Coder \
    --sens_a outputs/S_A.pt \
    --sens_b outputs/S_B.pt \
    --conflict_threshold 0.8 \
    --output_dir outputs/merged_model/
```
