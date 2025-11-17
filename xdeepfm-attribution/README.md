# xDeepFM Attribution Blueprint

本项目完全按照 `agents.md` 要求实现了 HuggingFace `criteo/criteo-attribution-dataset` 的 **可执行** 方案：提供 YAML/Markdown 说明，同时在 `src/xdeepfm` 下实现了完整的预处理、建模、训练与评估代码，可直接运行。

## 目录概览
- `configs/data/criteo_attribution.yaml`：数据源、特征、划分与预处理策略（时间 70/15/15 或随机 0.8/0.1/0.1）。
- `configs/model/xdeepfm.yaml`：xDeepFM 结构（embedding_dim=10、FM + DNN[400,400] + CIN[200,200,200]）。
- `configs/train/default.yaml`：训练超参（Adam lr=1e-3、early-stopping、logloss/AUC 指标）。
- `configs/experiment/xdeepfm_criteo_attr.yaml`：组合 data/model/train 并指定输出目录与种子。
- `src/data_pipeline.md` / `src/model_spec.md` / `src/training_flow.md`：对照 agents.md 第 4-7 节的自然语言指令。
- `src/entrypoints/*.md`：`run_preprocess`、`run_train`、`run_eval` 的具体执行步骤。

## 安装
```bash
pip install -r requirements.txt
```

## 使用顺序
1. **预处理**
   ```bash
   python -m tools.run_preprocess --config configs/data/criteo_attribution.yaml
   ```
2. **训练**
   ```bash
   python -m tools.run_train --experiment configs/experiment/xdeepfm_criteo_attr.yaml
   ```
3. **评估**
   ```bash
   python -m tools.run_eval --experiment configs/experiment/xdeepfm_criteo_attr.yaml \
     --checkpoint runs/xdeepfm_criteo_attr/checkpoints/best.pt
   ```

## 产物约定
- 所有日志、检查点与指标位于 `./runs/xdeepfm_criteo_attr/`。
- metadata/vocab/scaler 存放于 `./data/processed/criteo_attribution/`。

根据这些描述，任意 Agent 均可还原可执行的 xDeepFM 复现流程。
