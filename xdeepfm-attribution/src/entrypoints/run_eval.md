# 入口：run_eval

## 调用方式
```
python -m tools.run_eval \
  --experiment configs/experiment/xdeepfm_criteo_attr.yaml \
  --checkpoint runs/xdeepfm_criteo_attr/checkpoints/best.ckpt
```

## 主要逻辑
1. 解析 experiment YAML，确定数据、模型、训练配置与 `output_dir`。
2. 根据 metadata 构建 test_loader（或 CLI 指定的 split），保持原始时间顺序。
3. 载入 checkpoint，构建与训练阶段完全一致的 xDeepFM（embedding_dim=10、CIN/DNN 配置相同）。
4. 运行前向推理，计算 logloss、AUC，并可扩展 bucket 统计（cost / time_since_last_click）。
5. 将结果写入 `runs/xdeepfm_criteo_attr/metrics_test.yaml`，并生成 `preds_test.parquet`（包含 `uid, campaign, label, prediction`）。

## 依赖
- 训练生成的 checkpoint 与 metadata
- 与训练一致的模型/特征配置
