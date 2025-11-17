# 入口：run_train

## 运行位置
项目代码位于 `xdeepfm-attribution/`，可执行模块都在 `src/`。推荐的调用方式：
```bash
cd xdeepfm-attribution/src
python -m tools.run_train \
  --experiment ../configs/experiment/xdeepfm_criteo_attr.yaml
```
> `configs/` 在项目根目录；`load_yaml` 会自动解析当前工作目录或项目根目录下的相对路径，因此也可以继续传入 `configs/...`。

## 主要逻辑（`xdeepfm/train/pipeline.py::train_experiment`）
1. **加载实验配置**：`FullConfig.from_experiment` 解析 `experiment` YAML，拿到 data/model/train 配置路径、输出目录和 `seed` 并全部转成绝对路径。
2. **确保元数据可用**：调用 `_load_or_create_metadata`。若 `preprocess.output_dir/metadata.yaml` 不存在则直接触发 `run_preprocess`，否则复用缓存。
3. **构建数据加载器**：依据 `data.dataloader` 中的 `batch_size/num_workers/pin_memory/shuffle_train` 为 train/valid/test 创建 `torch.utils.data.DataLoader`，train split 支持按字典形式传入不同的 batch size。
4. **搭建模型**：`build_model` 读取模型 YAML（embedding_dim=10、FM+DNN[400,400]+CIN[200,200,200]），基于 metadata 中每个 categorical field 的 vocab size 构建独立 embedding，并将 dense/categorical 特征拼接后馈入 FM/DNN/CIN 三路，再汇总为单一 logit。
5. **训练配置**：
   - 设备：`training.device`（支持 `cpu`/`cuda`/`auto`，自动回退）。
   - 损失：`nn.BCEWithLogitsLoss`；推理时手动 `sigmoid` 计算概率。
   - 优化器：当前实现固定为 `torch.optim.Adam`，lr/weight_decay 来自 YAML（默认 `1e-3/1e-4`）。
   - 早停：`EarlyStopping(metric=valid_logloss, mode=min, patience=2, delta=1e-4)`；scheduler 配置暂未启用，保留扩展接口。
6. **训练循环**：按 `training.epochs`（默认 20）在 train/valid 上轮训；若监控指标改善则保存 `runs/xdeepfm_criteo_attr/checkpoints/best.pt` 并记录 `history`（包含 train/valid logloss 与 AUC）；若连续 `patience` 轮未提升则提前终止。
7. **测试与产物**：载入最优权重，在 test split 上计算 logloss/AUC，输出到 `runs/xdeepfm_criteo_attr/metrics_test.yaml`，并把含 `uid/campaign/label/prediction` 的预测写到 `runs/xdeepfm_criteo_attr/preds_test.parquet`；函数返回 best checkpoint 路径供后续 `run_eval` 使用。

## 依赖
- `configs/experiment/xdeepfm_criteo_attr.yaml`（自动引用 data/model/train YAML）
- 预处理产物：`data/processed/criteo_attribution/` 下的 `{split}.parquet` 与 `metadata.yaml`
- PyTorch、pandas、numpy、scikit-learn（`roc_auc_score`）
