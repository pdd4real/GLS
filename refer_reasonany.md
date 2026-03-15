# ReasonAny 代码仓库分析

> 论文: *ReasonAny: Incorporating Reasoning Capability to Any Model via Simple and Effective Model Merging*
> 代码位置: `refer/ReasonAny/`

---

## 第一部分: ReasonAny 在做什么

### 1.1 核心问题

现有的模型合并方法（Task Arithmetic、TIES、DARE、LED 等）在将 reasoning 模型与 domain-specific 模型合并时，会出现**性能坍塌（Performance Collapse）**——要么推理能力严重下降，要么领域能力被破坏，无法同时保持两者。ReasonAny 要解决的就是 "Reasoning + X" 的冲突问题。

### 1.2 核心发现

论文发现了一个**反直觉现象**：

- **推理能力**主要存储在模型中**梯度敏感度低**（low gradient sensitivity）的参数区域
- **领域能力**（safety / biomedicine / finance）主要对应**梯度幅值高**的参数区域

换言之，Long-CoT 推理能力对应的参数在 calibration data 上产生的梯度很小，而领域能力对应的参数梯度很大。这两类能力在参数空间中天然地占据不同区域。

### 1.3 方法论: Contrastive Gradient Identification

ReasonAny 的工作流程：

```
输入: base_model, reasoning_model (微调后), domain_model (微调后)
输入: reasoning_calibration_data, domain_calibration_data

Step 1: 计算 Task Vector
    task_vector = finetuned_model_params - base_model_params

Step 2: 用 calibration data 做前向 + 反向传播，得到每个参数的梯度重要性分数

Step 3: 参数选择
    - Domain 模型: 选梯度分数最高的 top k% 参数 (高梯度 = 领域能力所在)
    - Reasoning 模型: 选梯度分数最低的 bottom k% 参数 (低梯度 = 推理能力所在)

Step 4: Disjoint（互斥）
    - 将两组参数集合取差集，确保不重叠
    - T_domain_disjoint = T_domain - T_reasoning
    - T_reasoning_disjoint = T_reasoning - T_domain

Step 5: 合并
    - merged = base_model
    - merged[T_domain_disjoint] += scale_d * domain_task_vector
    - merged[T_reasoning_disjoint] += scale_r * reasoning_task_vector
```

### 1.4 实验覆盖范围

- **领域**: Safety（安全对齐）、Biomedicine（生物医学）、Finance（金融）
- **模型家族**: Qwen2.5（7B/14B/32B/QwQ-32B）、Llama 3.1-8B
- **对比方法**: 10 个 baseline（Task Arithmetic、LED、TIES、DARE 等）
- **评测**: 15+ benchmarks，超 1000 次实验

### 1.5 代码结构

```
ReasonAny/
├── ReasonAny/
│   ├── safety/          # Safety 领域合并 (main.py + run_safety.sh)
│   ├── biomedicine/     # 生物医学领域合并 (main.py + run_bio.sh)
│   └── finance/         # 金融领域合并 (main.py + run_finance.sh)
├── ablation_study/      # 消融实验 (ablation_reasonany.py)
├── eval/                # 评估脚本
│   ├── reasoning_bench/ # 推理评测 (GSM8K, MATH 等)
│   └── domain_bench/    # 领域评测
└── Figures/             # 论文图表
```

---

## 第二部分: Calibration Data 与反向传播深度分析

### 2.1 为什么要反向传播?

反向传播的目的**不是训练模型**，而是**探测每个参数对给定数据的敏感程度**。

具体而言，ReasonAny 需要回答一个问题：模型中的哪些参数负责推理能力，哪些参数负责领域能力？答案是通过梯度来判断的——对某类数据（如推理数据）做前向传播得到 loss，再反向传播得到每个参数的梯度。梯度的绝对值越大，说明该参数对这类数据越"敏感"（改动该参数会显著影响该类数据上的 loss）。

关键洞察：推理能力存在于**低梯度区域**，所以对 reasoning calibration data 做反向传播后，选取梯度最小的那些参数，就是推理能力的"藏身之处"。

### 2.2 反向传播在得到什么?

反向传播得到的是每个参数的**重要性分数（importance score）**，具体有两种计算方式：

#### 方式 1: Gradient Magnitude（默认，`is_gradient_base=True`）

```
importance_score[name] = |grad(param)|
```

直接取梯度绝对值。

#### 方式 2: SNIP Score（`is_gradient_base=False`）

```
importance_score[name] = |param.data * grad(param)|
```

参数值与梯度的乘积的绝对值，源自 SNIP（Single-shot Network Pruning）方法。

### 2.3 怎么得到的? —— 逐步流程

核心函数: `calculate_importance_scores()` （位于各 `main.py` 第 25-65 行）

```python
def calculate_importance_scores(model, tokenizer, texts, device, is_gradient_base=True):
    model.eval()  # 推理模式，但需要梯度

    # Step 1: 初始化每个参数的分数累加器为全零
    importance_scores = {name: torch.zeros_like(param)
                         for name, param in model.named_parameters()}

    for text in texts:  # 遍历每一条 calibration sample
        # Step 2: Tokenize，max_length=512
        inputs = tokenizer(text, return_tensors="pt",
                           max_length=512, truncation=True,
                           padding="max_length").to(device)
        labels = inputs["input_ids"]  # 自回归: labels = input_ids

        # Step 3: 前向传播，计算 causal LM loss
        model.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        # Step 4: 反向传播! 计算每个参数的梯度
        loss.backward()

        # Step 5: 累加梯度分数
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if is_gradient_base:
                        importance_scores[name] += torch.abs(param.grad)
                    else:
                        importance_scores[name] += torch.abs(param.data * param.grad)

    # Step 6: 取平均
    for name in importance_scores:
        importance_scores[name] /= len(texts)

    return importance_scores
```

流程总结：
1. 模型设为 `eval()` 模式（BN/Dropout 固定），但保留梯度计算
2. 对每条 calibration text，做一次完整的前向+反向传播
3. 取每个参数梯度的绝对值，累加到 importance_scores 中
4. 全部 samples 跑完后取平均，得到最终的 importance score

**注意**: 这个过程对每个模型（base_model、domain_model、reasoning_model）都需要分别执行。以 safety 场景为例，一共要做 **4 次** importance score 计算：
- `safety_model` + `safety_texts` -> safety 模型在 safety 数据上的梯度
- `base_model` + `safety_texts` -> base 模型在 safety 数据上的梯度
- `reasoning_model` + `reasoning_texts` -> reasoning 模型在 reasoning 数据上的梯度
- `base_model` + `reasoning_texts` -> base 模型在 reasoning 数据上的梯度

### 2.4 得到的分数怎么存储?

重要性分数**不持久化存储到磁盘**，全部在内存中以 Python dict 的形式保存：

```python
importance_scores = {
    "model.layers.0.self_attn.q_proj.weight": torch.Tensor(...),  # 与该参数同 shape
    "model.layers.0.self_attn.k_proj.weight": torch.Tensor(...),
    "model.layers.0.mlp.gate_proj.weight":    torch.Tensor(...),
    ...
    # 每个 named_parameter 都有对应的同 shape tensor
}
```

- **Key**: 参数名（`str`），如 `"model.layers.15.self_attn.v_proj.weight"`
- **Value**: 与参数同 shape 的 `torch.Tensor`，存储逐元素的重要性分数
- **生命周期**: 计算完参数选择后，通过 `del` + `gc.collect()` + `torch.cuda.empty_cache()` 显式释放

### 2.5 这些分数后来怎么用?

分数通过 `get_selected_param_names()` 函数转化为**参数名集合**，用于指导合并：

```python
# Step 1: 拼接所有分数为一维向量，计算 top/bottom k% 的阈值
all_scores = torch.cat([v.flatten() for v in scores.values()])
k = int(len(all_scores) * ratio)  # ratio 默认 0.05 (5%)

# Step 2: 计算阈值
if select_lowest:  # reasoning 用 bottom k%
    threshold = torch.kthvalue(all_scores, k).values
else:              # domain 用 top k%
    threshold = torch.kthvalue(all_scores, len(all_scores) - k).values

# Step 3: 筛选参数名（注意：是参数名级别，非逐元素）
selected_names = set()
for name, score_tensor in scores.items():
    if select_lowest:
        if (score_tensor <= threshold).any():  # 只要有一个元素 <= threshold
            selected_names.add(name)
    else:
        if (score_tensor >= threshold).any():  # 只要有一个元素 >= threshold
            selected_names.add(name)
```

**注意此处的粒度**: 虽然 importance score 是逐元素计算的，但参数选择是**按参数名（整个 tensor）**为单位的——只要一个 tensor 中有任何一个元素满足阈值条件，整个参数就被选中。

选出两组参数名后，进行 Disjoint（互斥）操作：

```python
T_domain_disjoint = T_domain - T_reasoning      # domain 独有的参数
T_reasoning_disjoint = T_reasoning - T_domain    # reasoning 独有的参数
```

最终合并：

```python
merged_state_dict = base_model.state_dict()
for name in T_domain_disjoint:
    merged_state_dict[name] += domain_scale * domain_task_vector[name]
for name in T_reasoning_disjoint:
    merged_state_dict[name] += reasoning_scale * reasoning_task_vector[name]
```

### 2.6 Calibration Data 的格式

所有 calibration data 均为 **Parquet** 格式文件，通过 `pd.read_parquet()` 加载。不同领域使用不同数据集和字段提取逻辑：

| 领域 | 数据集路径 | 字段提取逻辑 | 文本格式 |
|------|-----------|-------------|---------|
| **Reasoning** | `OpenThoughts-114k-math/data/train-00000-of-00005.parquet` | `conversations[0]['value'] + eos + conversations[1]['value']` | 问题 + EOS + 推理回答 |
| **Safety** | `hh-rlhf/train.parquet` | `row['chosen']` | 人类偏好的安全回答 |
| **Biomedicine** | `pubmedqa/train-00000-of-00001.parquet` | `row['question'] + eos + row['long_answer']` | 医学问题 + EOS + 详细回答 |
| **Finance** | `FinanceQA/0000.parquet` | `str(row['QUERY']) + eos + str(row['ANSWER'])` | 金融问题 + EOS + 回答 |

#### 各数据集详细格式：

**Reasoning (OpenThoughts-114k-math)**:
```json
{
  "conversations": [
    {"value": "数学问题文本..."},
    {"value": "详细推理解答..."}
  ]
}
```

**Safety (hh-rlhf)**:
```json
{
  "chosen": "完整的安全对话文本（人类偏好的回答）",
  "rejected": "被拒绝的回答（不使用）"
}
```

**Biomedicine (PubMedQA)**:
```json
{
  "question": "医学问题...",
  "long_answer": "详细的医学回答..."
}
```

**Finance (FinanceQA)**:
```json
{
  "QUERY": "金融问题...",
  "ANSWER": "金融回答..."
}
```

### 2.7 一共需要什么样的 Data?

完成一次完整的 ReasonAny 合并，需要准备以下数据：

| 数据类型 | 数量 | 用途 | 来源 |
|---------|------|-----|------|
| **Reasoning calibration data** | 100 条（默认 `NUM_SAMPLES=100`） | 计算 reasoning 模型和 base 模型的梯度重要性分数 | OpenThoughts-114k-math |
| **Domain calibration data** | 100 条（默认 `NUM_SAMPLES=100`） | 计算 domain 模型和 base 模型的梯度重要性分数 | hh-rlhf / PubMedQA / FinanceQA |

此外还需要三个**模型**（不是 data，但不可少）：

| 模型 | 说明 |
|------|-----|
| **Base Model** | 基座模型（如 Qwen2.5-7B） |
| **Reasoning Model** | 经过推理微调的模型（如 DeepSeek-R1-Distill） |
| **Domain Model** | 经过领域微调的模型（如 safety-tuned 模型） |

### 2.8 关键超参数

| 参数 | 默认值 | 含义 |
|------|-------|-----|
| `NUM_SAMPLES` | 100 | calibration 样本数 |
| `IS_GRADIENT_BASE` | True | True=梯度幅值, False=SNIP分数 |
| `IS_REASONING_TOP` | False | False=选低梯度参数(论文核心发现) |
| `SAFETY_RATIO` / `DOMAIN_RATIO` | 0.05 | 选择 top 5% 参数 |
| `REASONING_RATIO` | 0.05 | 选择 bottom 5% 参数 |
| `SAFETY_SCALE` / `DOMAIN_SCALE` | 1.0 | task vector 缩放因子 |
| `REASONING_SCALE` | 1.0 | reasoning task vector 缩放因子 |
| `IS_INTERSECT` | False (safety) | 是否对 finetuned 和 base 的分数取交集 |

### 2.9 整体数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Calibration Data Flow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Parquet Files ──load_texts()──> List[str] (100条文本)           │
│       │                              │                          │
│       │                    tokenize (max_len=512)               │
│       │                              │                          │
│       │                     model.forward()                     │
│       │                              │                          │
│       │                    CausalLM Loss                        │
│       │                              │                          │
│       │                    loss.backward()                      │
│       │                              │                          │
│       │               ┌──────────────┴──────────────┐           │
│       │               │                             │           │
│       │          |param.grad|              |param * param.grad|  │
│       │          (Gradient)                    (SNIP)            │
│       │               │                             │           │
│       │               └──────────────┬──────────────┘           │
│       │                              │                          │
│       │                   importance_scores                     │
│       │                   {name: Tensor}                        │
│       │                              │                          │
│       │               get_selected_param_names()                │
│       │                              │                          │
│       │                  Set[str] 参数名集合                     │
│       │                              │                          │
│       │                        Disjoint                         │
│       │                     ┌────┴────┐                         │
│       │              T_domain    T_reasoning                    │
│       │                  │           │                          │
│       │                  ▼           ▼                          │
│       │           merged_model[name] +=                         │
│       │              scale * task_vector[name]                  │
│       │                                                         │
│       │                 Save to Disk                            │
│       │          (HuggingFace format)                           │
└─────────────────────────────────────────────────────────────────┘
```
