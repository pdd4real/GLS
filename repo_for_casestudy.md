在深度学习和模型合并的语境下，**Case Study（案例分析）** 是连接宏观指标（如 MMLU 准确率）与微观黑盒机制的桥梁。单靠 Benchmark 分数只能证明方法“有效”，但跑通顶级会议级别的论文复现，还需要通过 Case Study 直观地向读者（或你自己）展示方法“为什么有效”以及“在何种极端边界下起作用”。它通常涉及对特定层、特定数据样本或特定注意力头的可视化与定性剖析。

结合你的项目方案，下面为你细化 GLS 方法的 Case Study，并将其无缝集成到现有的工程架构中。

---

### 1. GLS 方法 Case Study 细化方案

根据《GLS_Merging_Proposal.md》中的 Phase 3 规划，我们可以将理论性的验证转化为在 VS Code 中可直接执行的观测实验。我们将聚焦于“动态冲突解决”与“任务特定子空间（Task-Specific Subspaces）”的隔离效果。

#### A. 冲突可视化 (Visualizing Conflict & Routing)

仅仅计算出 $\alpha$ 并不够，我们需要直观看到专家模型是如何“瓜分”网络结构的。

* **Layer-wise 热力图**：提取所有 Transformer 层的 $S_{layer\_A}$ 和 $S_{layer\_B}$，绘制相对敏感度热力图。目标是观察数学专家模型是否在特定的 MLP 层出现极端的激活峰值，从而印证“不同技能驻留在不同深度”的假设。
* **动态粒度回放**：提取 $\text{Conflict}(A, B) \ge 0.8$ 的层，单独绘制这些层内部的 Channel-wise 权重分配图。通过这个 Case，证明在两模型争夺主导权时，细粒度合并成功避开了特征中和。

#### B. 技能驻留与注意力探针 (Skill Preservation Probing)

合并模型最怕的是“灾难性遗忘”或通道间的特征污染。我们需要验证合并后的模型是否还保留着原始专家的特征空间。

* **对比激活模式**：输入一道 GSM8K 的数学题和一道 HumanEval 的代码题。使用 PyTorch Hook 拦截合并模型在特定“数学头 (Math Head)”和“代码头 (Code Head)”上的 Attention Map。
* **定性验证**：如果 GLS 方法有效，合并模型在处理数学题时，其激活路径应该高度拟合原始的 Expert A；在处理代码时，高度拟合 Expert B。这证明了合并操作没有破坏底层的结构协方差。

#### C. 极端失效分析 (Failure Mode Analysis)

科学的 Case Study 也应包含对局限性的探讨。

* 选取校准集（`calib_mixed_150.jsonl`）中导致预测崩溃的极少数 Out-of-Distribution (OOD) 样本。分析是否因为引入了熵正则化（Entropy Regularization），导致模型在某些罕见任务上被过度平滑。

---

### 2. 代码仓库结构更新

为了支持上述 Case Study，我们需要在《repo_build.md》原有的 `GLS-Merging/` 目录下 增加专门的分析模块。原则是**高内聚、低耦合**：不改动核心的前向和合并逻辑，仅仅增加可视化工具和探针脚本，方便你在远程服务器上生成图表或拦截张量。

```text
GLS-Merging/
├── data/
│   └── calib_mixed_150.jsonl        
├── src/
│   ├── data_loader.py               
│   ├── hooks.py                     
│   ├── sensitivity.py               
│   ├── merge_methods/
│   │   ├── layer_wise.py            
│   │   ├── channel_wise.py          
│   │   └── gls_router.py            
│   └── utils/
│       ├── model_io.py              
│       └── visualizer.py            # [新增] 存放 matplotlib/seaborn 绘图函数，用于渲染 Heatmap
├── scripts/
│   ├── prepare_calib_data.py        
│   ├── run_activation_cache.py      
│   └── run_merge.py                 
├── case_studies/                    # [新增] 专门用于跑 Phase 3 案例分析的独立目录
│   ├── plot_conflict_heatmap.py     # [新增] 读取暂存的 S 矩阵，绘制 Layer/Channel 敏感度分布图
│   ├── probe_attention_maps.py      # [新增] 针对 GSM8K/HumanEval 样本，挂载 Hook 提取并对比 Attention 激活状态
│   └── analyze_extreme_layers.py    # [新增] 专门输出 Conflict Rate >= 0.8 层级的详细混合比例日志，用于论文图表支撑
├── .vscode/
│   └── launch.json                  
├── requirements.txt                 # 需要补充 matplotlib, seaborn 等分析绘图库
└── README.md

```

增加 `case_studies` 目录的优势在于，当你完成 MVP 跑通后，可以直接把合并好的模型路径传给这些脚本，它们会复用 `src/hooks.py` 中的拦截机制来输出科研级别的对比图表。
