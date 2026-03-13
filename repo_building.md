为了让你能迅速、顺利地在服务器上跑通 GLS-Merging 的 MVP（最小可行性产品）版本，我们需要将理论转化为极具可操作性的工程步骤。

考虑到高效开发和后续扩展，这份细化方案将围绕 **PyTorch** 生态构建，并专门针对你在使用 VS Code 远程连接 GPU 实例进行代码调试和论文复现时的工程习惯进行了结构优化。

以下是为你定制的 `[模型 - 数据集 - 策略与实现]` 详细代码仓库搭建 Proposal：

---

### 1. 模型选择与环境基座 (Model & Environment)

为了确保架构一致性，避免词表（Vocabulary）或层结构不同导致的合并崩溃，基础模型和专家模型必须属于同一家族。

* **Base Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
* **Expert A (Math)**: 选取基于 Llama-3-8B 微调的数学开源模型（例如 HuggingFace 上的 `RLHFlow/LLaMA3-iterative-DPO-final` 或社区的 Llama-3-8B-Math 版本）。
* **Expert B (Code)**: 选取基于 Llama-3-8B 微调的代码模型（例如 `TechxGenus/Meta-Llama-3-8B-Instruct-Coder`）。
* **计算目标**: 确保单张 24GB/40GB 显存的 GPU 就能完成整个合并流程（这依赖于我们后面的 Activation Proxy 策略）。

---

### 2. 数据集：校准数据获取策略 (Data & Calibration)

**策略目标**：用极小的数据量（约 100-200 条）激活模型，捕获能代表 Task-Specific 子空间的特征。

* **获取策略**：采用 **"公开代理数据采样 (Public Proxy) + 领域均衡混合"** 策略。为了追求 MVP 跑通速度，先不搞模型自生成（Synthetic），直接从现成的高质量数据集中截取。
* **具体实现步骤**：
1. **代码域 (Code)**：从 HuggingFace 加载 `openai_humaneval` 数据集，随机抽取 50 条 Python 编程指令。
2. **数学域 (Math)**：加载 `gsm8k` (main 验证集)，随机抽取 50 条数学应用题。
3. **通用域 (General)**：从 `tatsu-lab/alpaca` 中抽取 50 条日常问答。
4. **数据固化**：在代码仓库中写一个 `scripts/prepare_calib_data.py`，将这 150 条数据统一格式化为 Llama-3 的 Chat Template，并保存为本地的 `data/calib_mixed_150.jsonl`。后续所有实验固定读取此文件，确保对比变量单一。



---

### 3. 核心计算：Activation Proxy 近似策略

**策略目标**：解决 8B 模型算梯度容易引发 OOM（显存溢出）的问题，用前向传播的激活值（Activation）代替反向传播的梯度（Gradient），大幅降低计算成本。

* **理论依据**：借鉴 Wanda 等剪枝算法的经验，参数的重要性可以近似为**权重幅值 $\times$ 输入激活值的二范数**：

$$S_{approx} = |W| \cdot ||X||_2$$


* **具体实现步骤 (PyTorch Hook 机制)**：
1. **挂载 Hook**：使用 PyTorch 的 `register_forward_hook`，遍历模型的所有层（如 `q_proj`, `v_proj`, `mlp.down_proj`）。
2. **前向推理**：将 `calib_mixed_150.jsonl` 分批次输入 Expert A 和 Expert B 进行**纯前向传播 (Forward Pass Only, `with torch.no_grad():`)**。
3. **捕获与累加**：在 Hook 函数中，拦截每个线性层的输入激活张量 $X$。计算其在 Sequence 维度上的 L2 范数 $||X||_2$，并累加这 150 条数据的平均激活强度。
4. **计算敏感度**：将累加好的激活强度与当前层的权重绝对值 $|W|$ 相乘，得到该层每个参数的近似敏感度矩阵 $S$。这步计算完全在内存/显存中原地进行，算完即释放激活值，显存峰值仅相当于单次 Batch Size 推理的开销。



---

### 4. 融合执行：Coarse-to-Fine 动态粒度策略

**策略目标**：既要保留 Transformer 层级的协方差结构（粗粒度），又要能在模型发生能力冲突时，下探到通道/头级别解决冲突（细粒度）。

* **判断指标 (Conflict Rate)**：使用相对差异来衡量层级冲突。如果 $S_A$ 和 $S_B$ 在某一层非常接近，说明两个模型都在争夺该层的主导权，此时粗粒度合并会互相干扰。

$$\text{Conflict}(A, B) = 1 - \frac{|S_{layer\_A} - S_{layer\_B}|}{S_{layer\_A} + S_{layer\_B}}$$



Conflict 越接近 1，冲突越大。
* **具体实现步骤**：
1. **计算 Layer-wise 敏感度**：对上一步得到的敏感度矩阵 $S$，求该层所有元素的总和，得到标量 $S_{layer\_A}$ 和 $S_{layer\_B}$。
2. **动态判定**：设定一个超参数阈值（例如 $\theta_{conflict} = 0.8$）。
* **If Conflict < 0.8 (低冲突)**：执行 **Layer-wise 合并**。直接用 $S_{layer\_A}$ 和 $S_{layer\_B}$ 算出一个全局的 $\alpha_A$，该层的所有参数统一按 $W_{new} = \alpha_A W_A + \alpha_B W_B$ 混合。
* **If Conflict $\ge$ 0.8 (高冲突)**：下探执行 **Channel-wise (或 Head-wise) 合并**。对于 MLP 层，按列（Output Channel）将敏感度矩阵 $S$ 切片，分别计算每一列的 $\alpha_{channel\_i}$，并对权重矩阵按列进行不同比例的混合。


3. *(进阶预留)*：在这部分代码中，可以预留出之前我们讨论过的“正交子空间投影”或“带温度的 Softmax”正则化函数的接口，方便跑通 MVP 后立刻进行消融实验。



---

### 5. 代码仓库架构建议 (Repository Structure)

为了便于你在 VS Code 中进行模块化调试和管理，建议采用以下结构：

```text
GLS-Merging/
├── data/
│   └── calib_mixed_150.jsonl        # 固化的校准数据集
├── src/
│   ├── data_loader.py               # 负责加载和 tokenize 校准数据
│   ├── hooks.py                     # 核心！存放 register_forward_hook 逻辑，提取 Activation
│   ├── sensitivity.py               # 计算 S_approx 以及 Conflict Rate
│   ├── merge_methods/
│   │   ├── layer_wise.py            # 粗粒度合并逻辑
│   │   ├── channel_wise.py          # 细粒度合并逻辑
│   │   └── gls_router.py            # 根据 Conflict Rate 动态调用 coarse-to-fine 的路由
│   └── utils/
│       └── model_io.py              # 高效加载/保存 HuggingFace 权重的工具
├── scripts/
│   ├── prepare_calib_data.py        # 数据集下载与处理脚本
│   ├── run_activation_cache.py      # 第一阶段：跑前向传播，缓存敏感度 S 到本地
│   └── run_merge.py                 # 第二阶段：读取 S，执行合并，保存新模型
├── .vscode/
│   └── launch.json                  # 配置 debug 参数，方便直接在具体脚本打断点
├── requirements.txt
└── README.md

```

**工程小贴士**：
在跑通 MVP 时，建议将“前向传播提取敏感度”和“基于敏感度合并权重”拆分成两个独立的脚本（如上图 `scripts` 目录所示）。
将提取出来的 $S_A$ 和 $S_B$ 存成 `.pt` (PyTorch Tensor) 文件。这样在后续你疯狂调整 Coarse-to-Fine 阈值或尝试 Softmax 熵正则化公式时，**不需要重新跑耗时的模型前向推理**，只需几秒钟就能在内存中完成新的权重合并，大幅提升迭代效率。