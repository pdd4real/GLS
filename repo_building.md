既然算力储备充足，我们就可以彻底抛弃那些为了妥协显存而设计的近似算法（如 Activation Proxy），**直接回归 GLS-Merging 最硬核、最精确的数学本质：真实梯度计算 (True Gradient Calculation)**。

在“显卡管够”的前提下，这份 Proposal 将升级为“满血版”。我为你保留了无关的架构和数据策略，仅对核心计算模块（第 3 部分）和环境设定（第 1、5 部分）进行了针对性的“算力解放”修改：

---

### 1. 模型选择与环境基座 (Model & Environment)

* **Base Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
* **Expert A (Math)**: 基于 Llama-3-8B 微调的数学模型（如 `RLHFlow/LLaMA3-iterative-DPO-final`）。
* **Expert B (Code)**: 基于 Llama-3-8B 微调的代码模型（如 `TechxGenus/Meta-Llama-3-8B-Instruct-Coder`）。
* **[修改点] 算力部署设定**: 由于需要对 8B 模型进行完整的反向传播（Backward Pass），单卡 24GB 将无法满足。我们将直接采用 **Multi-GPU (如 4x A100 或 8x 4090)** 环境，结合 HuggingFace `Accelerate` 或 `DeepSpeed Zero-2/3` 将模型权重和梯度分片加载，确保真实梯度的无损计算。

---

### 2. 数据集：校准数据获取策略 (Data & Calibration)

*(本部分保持不变，依然采用最稳健的公开代理数据采样)*

* **获取策略**：采用 **"公开代理数据采样 (Public Proxy) + 领域均衡混合"** 策略。
* **具体实现步骤**：
1. **代码域 (Code)**：从 `openai_humaneval` 随机抽取 50 条指令。
2. **数学域 (Math)**：从 `gsm8k` 抽取 50 条。
3. **通用域 (General)**：从 `alpaca` 抽取 50 条。
4. **数据固化**：统一格式化为 Llama-3 的 Chat Template，保存为本地的 `data/calib_mixed_150.jsonl`。



---

### 3. [核心重构] 计算：True Gradient Sensitivity 真实梯度计算策略

**策略目标**：既然显卡管够，直接硬算模型在校准集上的真实一阶导数（或二阶 Fisher 近似），获取最精确的参数重要性指标 $S_{m} = \sum | \nabla_{\theta} \mathcal{L}(x) \cdot \theta |$。

* **具体实现步骤 (PyTorch Backward 机制)**：
1. **分布式加载与梯度开启**：使用 `Accelerate` 加载 Expert A 和 B，并将所有参与合并的层（Attention, MLP）的 `requires_grad` 设为 `True`。
2. **前向与反向传播 (Forward & Backward)**：将 `calib_mixed_150.jsonl` 分 Batch 输入模型。计算标准的 Causal Language Modeling 损失（即 Next-token prediction 的 CrossEntropy Loss）。
3. **执行反向传播**：调用 `loss.backward()`。此时，PyTorch 会自动在每一层的 `param.grad` 中存储真实的梯度矩阵 $\nabla_{\theta} \mathcal{L}(x)$。
4. **敏感度捕获与累加**：
* 遍历模型的 `named_parameters()`。
* 提取权重 `W = param.data` 和梯度 `G = param.grad`。
* 计算当前 Batch 的敏感度矩阵：$S_{batch} = \text{torch.abs}(G \times W)$。（*注：如果想用 Fisher 近似，改为 $S_{batch} = (G^2) \times \text{torch.abs}(W)$*）。
* 将 $S_{batch}$ 累加到全局的 $S_{total}$ 中。


5. **清理梯度**：关键一步！每次累加完后，必须调用 `model.zero_grad()` 清空显存中的梯度图，防止下一个 Batch OOM，如此循环直到 150 条数据跑完。



---

### 4. 融合执行：Coarse-to-Fine 动态粒度策略

*(本部分逻辑不变，但此时输入的 $S$ 是最精准的真实梯度敏感度)*

* **判断指标 (Conflict Rate)**：

$$\text{Conflict}(A, B) = 1 - \frac{|S_{layer\_A} - S_{layer\_B}|}{S_{layer\_A} + S_{layer\_B}}$$


* **具体实现步骤**：
1. **计算 Layer-wise 敏感度**：对真实敏感度矩阵 $S$ 求和，得到 $S_{layer\_A}$ 和 $S_{layer\_B}$。
2. **动态判定**：设定阈值（例如 $\theta_{conflict} = 0.8$）。
* **低冲突 (Conflict < 0.8)**：执行 **Layer-wise 合并**。
* **高冲突 (Conflict $\ge$ 0.8)**：下探执行 **Channel-wise (或 Head-wise) 合并**。按列（Output Channel）切片敏感度矩阵，分别计算并混合权重。





---

### 5. [调整] 代码仓库架构建议 (Repository Structure)

去掉了原先的 `hooks.py`（因为不需要强行 Hook 激活值了），改为标准的梯度计算引擎。

```text
GLS-Merging/
├── data/
│   └── calib_mixed_150.jsonl
├── src/
│   ├── data_loader.py
│   ├── gradient_engine.py           # [修改] 核心！负责 loss.backward()、提取 param.grad 和累加 S
│   ├── sensitivity.py               # 计算 Conflict Rate
│   ├── merge_methods/
│   │   ├── layer_wise.py
│   │   ├── channel_wise.py
│   │   └── gls_router.py
│   └── utils/
│       └── model_io.py              # 需增加对 Accelerate/DeepSpeed 分布式权重的支持
├── scripts/
│   ├── prepare_calib_data.py
│   ├── run_gradient_cache.py        # [修改] 跑完整前向+反向传播，提取真实梯度敏感度 S 并固化为 .pt
│   └── run_merge.py                 # 读取 S 的 .pt 文件，在 CPU/单卡上光速完成权重矩阵混合
├── .vscode/
│   └── launch.json
├── requirements.txt
└── README.md

```

**工程小贴士 (针对算力充足环境)**：
哪怕你有 8 张 A100，我也强烈建议保留 `run_gradient_cache.py` 和 `run_merge.py` **分离**的设计。
算力用来暴力跑完 150 条数据的反向传播，将极其珍贵的真实梯度敏感度 $S_A$ 和 $S_B$ 存成 `.pt` 文件。后续哪怕你要测试 100 种不同的 Coarse-to-Fine 阈值或 Softmax 温度系数，也只需要用 CPU 跑 `run_merge.py` 即可，**秒级出结果，绝不浪费昂贵的 GPU 算力去重复算梯度**。

接下来，需要我为你把 `gradient_engine.py` 中最关键的 `loss.backward()` 提取梯度并累加的代码框架写出来吗？