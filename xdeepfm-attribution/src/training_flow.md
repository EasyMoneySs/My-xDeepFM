# 训练流程说明

该文档覆盖 `agents.md` 第 6、7 节要求，约束 `run_preprocess`、`run_train`、`run_eval` 的核心逻辑。

## 6.1 训练入口 `run_train`
1. 读取 `configs/experiment/xdeepfm_criteo_attr.yaml`，解析 data/model/train 配置路径。
2. 依赖数据配置：
   - 若 `preprocess.output_dir` 不存在或 metadata 不匹配，先触发 `run_preprocess`。
   - 构建 Train/Valid/Test DataLoader（train 打乱，valid/test 保留顺序）。
3. 依赖模型配置：
   - 基于 metadata 中的 `field_vocab_sizes`、numerical 维度实例化 xDeepFM（FM + DNN + CIN）。
   - 初始化 embedding、CIN、DNN 等参数。
4. 依赖训练配置：
   - 设置 optimizer (Adam)、lr_scheduler、loss (BCE) 与 metrics (logloss/AUC)。
   - 进入 `epochs=20` 的循环：train 阶段前向/反向；valid 阶段计算指标、根据 `early_stopping` 决定是否保存 checkpoint/提前停止。
5. 训练完成后加载最佳模型，在 test_loader 上评估，将 logloss/AUC 写入 `experiment.output_dir`（如 `runs/xdeepfm_criteo_attr/metrics_test.yaml`），并产出预测文件。

## 6.2 评估指标定义
- **Logloss**：二分类交叉熵。
- **AUC**： ROC AUC。
- 可选统计：正负样本比例、cost 或 time_since_last_click bucket 的 AUC（写入 summary）。

## run_preprocess & run_eval 补充
- `run_preprocess`：实现 `src/data_pipeline.md` 描述的所有步骤（加载 → 划分 → 类别编码 → 数值处理 → 元数据写入）。
- `run_eval`：加载 experiment YAML + checkpoint，在指定 split（默认 test）上推理，输出指标与 `preds_test.parquet`。

## 7. 执行顺序 checklist
1. **预处理阶段**：读取数据配置 → HuggingFace 下载 → 按时间/随机切分 → 处理 categorical/numerical → 保存 processed 数据 + vocab/scaler/metadata。
2. **模型构建阶段**：读取 `configs/model/xdeepfm.yaml` → 用 metadata 创建 embedding、FM、DNN、CIN 分支 → 准备输出层。
3. **训练阶段**：加载 `configs/train/default.yaml` → 配置 optimizer/loss/metrics → 训练并根据 early-stopping 保存最佳模型。
4. **测试与结果保存**：最佳 checkpoint 在 test 集上计算 logloss/AUC → 将结果写入 `runs/xdeepfm_criteo_attr/metrics_test.yaml` 并记录关键超参与划分方式。
