# GLS-Merging

基于 Activation Proxy 近似 + Coarse-to-Fine 动态粒度的 LLM 模型合并框架。

## 流程

```
Stage 1: 提取敏感度
  python scripts/prepare_calib_data.py          # 准备校准数据
  python scripts/run_activation_cache.py ...    # 前向推理，缓存 S_A / S_B / conflict_rates

Stage 2: 执行合并
  python scripts/run_merge.py ...               # 读取缓存，合并权重，保存模型
```

## 目录结构

```
GLS-Merging/
├── data/                          # 校准数据与敏感度缓存（.pt）
├── src/
│   ├── data_loader.py             # 数据加载与 tokenize
│   ├── hooks.py                   # forward hook，提取激活统计
│   ├── sensitivity.py             # S_approx 与 Conflict Rate 计算
│   ├── merge_methods/
│   │   ├── layer_wise.py          # 粗粒度合并
│   │   ├── channel_wise.py        # 细粒度合并
│   │   └── gls_router.py          # 动态路由（Coarse-to-Fine）
│   └── utils/
│       └── model_io.py            # 模型/敏感度 I/O
├── scripts/
│   ├── prepare_calib_data.py
│   ├── run_activation_cache.py
│   └── run_merge.py
└── .vscode/launch.json            # VSCode 调试配置
```

## 安装

```bash
pip install -r requirements.txt
```
