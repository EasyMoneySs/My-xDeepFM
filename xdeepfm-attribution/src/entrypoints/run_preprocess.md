# 入口：run_preprocess

## 调用方式
```
python -m tools.run_preprocess \
  --config configs/data/criteo_attribution.yaml
```

## 主要逻辑
1. 解析数据配置（dataset/task/features/preprocess/dataloader）。
2. 使用 `datasets.load_dataset("criteo/criteo-attribution-dataset")`，缓存到 `./cache/hf_datasets`。
3. 根据 `preprocess.split.strategy`：
   - `time`：按 `timestamp` 排序，使用 70%/15%/15% 时间百分位切分。
   - `random`：按照 0.8/0.1/0.1 比例+`seed=42` 随机划分。
4. 类别特征处理：统计频次，`min_freq=10` 以下映射到 `__OOV__`，其余从 1 开始编号并保存 vocab。
5. 数值特征处理：`fill_na=0.0`、`scaler=standard`，并根据 `time_features` 生成 `delta_conversion_time`、`day_index`、`hour_in_day`。
6. 将各 split 写入 `./data/processed/criteo_attribution/{train,valid,test}.parquet`，同时输出 vocab、scaler、split 阈值等 metadata。

## 依赖
- `configs/data/criteo_attribution.yaml`
- HuggingFace Datasets
- pandas / pyarrow（或其他支持 Parquet 的库）
