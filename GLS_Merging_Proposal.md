# Proposal: GLS-Merging (Gradient-guided Layer-wise Sensitivity Merging)

## 1. 背景与动机 (Background & Motivation)

现有的 Model Merging 方法（如 TIES, DARE）虽然有效，但存在两个主要局限：
1.  **Magnitude $\neq$ Importance**: 仅通过权重幅值 ($|w|$) 判断重要性是不准确的。某些参数虽然变动小，但对特定功能至关重要（高梯度敏感度）。
2.  **Parameter-wise Destroys Structure**: 独立的参数级合并破坏了 Transformer 内部矩阵（如 Attention Head）的协同效应 (Covariance)。

**核心理念**: 引入 **Gradient Sensitivity (梯度敏感度)** 作为重要性指标，并坚持 **Structure-Aware (Layer/Head-wise)** 的合并粒度。

---

## 2. 方法概述 (Methodology)

GLS-Merging 包含三个关键步骤：

### 2.1 敏感度计算 (Sensitivity Calculation)
使用极少量校准数据 ($D_{cal}$)，计算每个模型参数的功能敏感度。
$$ S_{m}^{(l, c)} = \sum_{x \in D_{cal}} \left| \nabla_{\theta} \mathcal{L}(x) \cdot \theta \right| $$
*   **Granularity**: 聚合到 **Channel / Head** 级别，而非 Parameter 级别。

### 2.2 动态冲突解决 (Adaptive Conflict Resolution)
根据敏感度比例动态决定合并系数 $\alpha$：
$$ \alpha_A = \frac{S_A}{S_A + S_B} $$
*   **Advantage**: 自动识别“哪个模型更懂这一层/这一头”，无需人工调整超参。

### 2.3 粒度选择 (Granularity Selection)
放弃 Parameter-wise，选择 **Head-wise / Channel-wise**。
*   **Reason**: 这是 Transformer 的最小功能单元。既保留了矩阵内部结构，又提供了足够的灵活性来区分不同技能。

---

## 3. 挑战与解决方案 (Challenges & Solutions)

我们在初步构思中识别了以下挑战，并制定了应对方案：

| Challenge (挑战) | Proposed Solution (解决方案) |
| :--- | :--- |
| **Data Dependency**<br>(依赖数据) | **Synthetic Calibration**: 使用模型自生成的 Response 作为校准数据（Gradient Direction 一致性）。<br>**Public Proxy**: 使用微量公开数据 (Pile/C4) 代理。 |
| **Computational Cost**<br>(计算成本高) | **Sequential Calculation**: 逐层计算梯度并卸载，显存峰值 $\approx$ 单层推理。<br>**Forward Proxy**: 使用 Activation Magnitude 作为梯度的快速近似。 |
| **Granularity Trade-off**<br>(粒度难以平衡) | **Coarse-to-Fine Strategy**: 先算 Layer-wise，若冲突高（两模型都抢这一层）则下探到 Head-wise，否则直接 Layer-wise 平均。 |
| **OOD Robustness**<br>(过拟合校准集) | **Entropy Regularization**: 惩罚极端的 $\alpha$ (0 或 1)，强制保留一部分 Base 能力。<br>**Mixed Calibration**: 混合多种任务数据 (Code, Math, General) 避免偏科。 |

---

## 4. 实验验证计划 (Experimental Plan & Todo List)

为验证 GLS 的有效性，我们需要执行以下实验和 Case Study。


### Phase 0: Hypothesis Verification (Pre-experiments)
**验证我们的两个核心假设是否成立**：

#### 验证 1: Magnitude $\neq$ Importance
- [ ] **Correlation Analysis**: 计算权重幅值 ($|W|$) 与 梯度敏感度 ($|W \cdot G|$) 之间的 Spearman 相关系数。如果相关性低，说明 Magnitude 是一个糟糕的 proxy。
- [ ] **Ablation Pruning**:
    -   Top-K Magnitude Pruning: 砍掉幅值最小的 20%，测试 MMLU 下降多少。
    -   Top-K Sensitivity Pruning: 砍掉敏感度最小的 20%，测试 MMLU 下降多少。
    -   **预期结果**: Sensitivity Pruning 后的模型性能应显著优于 Magnitude Pruning（即它更准确地识别了“无用参数”）。

#### 验证 2: Parameter-wise Destroys Structure
- [ ] **Feature Map CKA (Centered Kernel Alignment)**:
    -   计算 Merged Model 与 Original Expert 在各层输出特征上的相似度 (CKA Score)。
    -   对比组: Parameter-wise Averaging vs. Layer-wise Averaging。
    -   **预期结果**: Layer-wise Merging 的 CKA Score 应高于 Parameter-wise，证明其更好地保留了原始特征空间的结构。
- [ ] **Attention Map Consistency**:
    -   可视化 Attention Head 的激活模式。检查 Parameter-wise 合并是否导致 Attention Map 变得杂乱无章（熵增），而 Layer/Head-wise 合并则保持了清晰的关注点。

### Phase 1: Feasibility & Baseline (可行性与基线)

- [ ] **Implementation**:而在 `merkit` 或手动脚本中实现 Layer-wise Gradient Calculation。
- [ ] **Data Setup**:
    -   构建 `Calib-Set-100`: 包含 100 条多样的 Instruction。
    -   构建 `Calib-Synthetic`: 让模型自己生成 100 条数据。
- [ ] **Baseline Comparison**:
    -   Models: Llama-3-8B-Instruct (Base) + WizardMath (Expert A) + CodeLlama (Expert B)。
    -   Baselines: Simple Averaging, Task Arithmetic, TIES-Merging, DARE.
    -   Target: GLS-Merging (Layer-wise & Head-wise)。
    -   Metrics: MMLU (General), GSM8K (Math), HumanEval (Code)。

### Phase 2: Ablation Studies (消融实验)
- [ ] **Granularity Check**: 对比 Parameter-wise vs. Head-wise vs. Layer-wise 的性能与显存开销。
- [ ] **Data Source Impact**: 对比 Real Data vs. Synthetic Data 对最终效果的影响。
- [ ] **Metric Choice**: 对比 Gradient ($|g \cdot w|$) vs. Activation ($|x|$) vs. Weight ($|w|$) 作为重要性指标的区别。

### Phase 3: Case Studies & Visualization (案例分析)
- [ ] **Visualizing Conflict**:
    -   画出 Layer-wise Sensitivity Heatmap。观察 Math 模型是否在某些特定层（如 MLP layers）表现出极高的敏感度？
    -   验证 GLS 是否正确地在这些层给予了 Math 模型更高的权重。
- [ ] **Skill Preservation**:
    -   通过 probing 检查合并后的模型在“数学 Head”和“代码 Head”上的激活模式是否被保留。

### Phase 4: Scalability (扩展性验证)
- [ ] **Merging Many Models**: 尝试合并 5-10 个不同领域的模型，测试 Entropy Regularization 是否有助于稳定合并过程。
