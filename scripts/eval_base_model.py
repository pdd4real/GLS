"""
功能：评估 base model 在 GSM8K（完整测试集）和 HumanEval（pass@1）上的 accuracy。

运行方式：
    python scripts/eval_base_model.py \
        --model  meta-llama/Meta-Llama-3-8B-Instruct \
        --tasks  gsm8k humaneval \
        --output results/base_model_eval.json

命令行参数：
    --model   : str  — 模型路径或 HuggingFace Hub ID
    --tasks   : list — 评估任务，可选 gsm8k / humaneval，默认两个都跑
    --output  : str  — 结果 JSON 保存路径，默认 results/base_model_eval.json
    --batch_size : int — 推理 batch size，默认 8
    --max_new_tokens : int — 生成最大 token 数，默认 512
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def extract_answer_gsm8k(text: str) -> str | None:
    """
    从模型输出中提取最终数字答案。
    GSM8K 标准格式：答案出现在 "#### <number>" 之后，或最后一个出现的数字。

    输入：
        text : str — 模型生成的完整文本

    输出：
        str | None — 提取到的数字字符串（去除逗号），提取失败返回 None
    """
    # 优先匹配 "#### 1234" 格式
    match = re.search(r"####\s*([\d,\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # 回退：取最后一个整数
    numbers = re.findall(r"-?\d[\d,]*", text)
    return numbers[-1].replace(",", "") if numbers else None


def eval_gsm8k(model, tokenizer, batch_size: int, max_new_tokens: int) -> dict:
    """
    在 GSM8K 完整测试集上评估模型，返回 accuracy。

    输入：
        model          : PreTrainedModel
        tokenizer      : PreTrainedTokenizer
        batch_size     : int
        max_new_tokens : int

    输出：
        dict — {"accuracy": float, "correct": int, "total": int}
    """
    import torch
    from datasets import load_dataset
    from tqdm import tqdm

    dataset = load_dataset("gsm8k", "main", split="test")
    correct = 0
    total = len(dataset)

    for i in tqdm(range(0, total, batch_size), desc="GSM8K"):
        batch = dataset[i: i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in batch["question"]
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # 只解码新生成的部分
        gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for pred, gold_answer in zip(preds, batch["answer"]):
            gold = extract_answer_gsm8k(gold_answer)
            pred_ans = extract_answer_gsm8k(pred)
            if gold is not None and pred_ans == gold:
                correct += 1

    return {"accuracy": correct / total, "correct": correct, "total": total}


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

def eval_humaneval(model, tokenizer, max_new_tokens: int) -> dict:
    """
    在 HumanEval 上评估 pass@1（greedy decoding，每题生成 1 个样本）。
    使用 human_eval 官方包执行代码并判断通过率。

    输入：
        model          : PreTrainedModel
        tokenizer      : PreTrainedTokenizer
        max_new_tokens : int

    输出：
        dict — {"pass@1": float, "total": int}
    """
    import torch
    from datasets import load_dataset
    from human_eval.data import write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
    from tqdm import tqdm

    dataset = load_dataset("openai_humaneval", split="test")
    samples = []

    for item in tqdm(dataset, desc="HumanEval"):
        prompt = item["prompt"]
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Complete the following Python function:\n\n{prompt}"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = output[0, inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # human_eval 期望 completion 是函数体（不含 prompt）
        samples.append({"task_id": item["task_id"], "completion": completion})

    # 写临时文件，调用官方评估
    tmp_path = "/tmp/humaneval_samples.jsonl"
    write_jsonl(tmp_path, samples)
    results = evaluate_functional_correctness(tmp_path)
    return {"pass@1": results["pass@1"], "total": len(samples)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "humaneval"],
                        choices=["gsm8k", "humaneval"])
    parser.add_argument("--output", default="results/base_model_eval.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    results = {"model": args.model}

    if "gsm8k" in args.tasks:
        print("\n=== GSM8K ===")
        results["gsm8k"] = eval_gsm8k(model, tokenizer, args.batch_size, args.max_new_tokens)
        print(results["gsm8k"])

    if "humaneval" in args.tasks:
        print("\n=== HumanEval ===")
        results["humaneval"] = eval_humaneval(model, tokenizer, args.max_new_tokens)
        print(results["humaneval"])

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
