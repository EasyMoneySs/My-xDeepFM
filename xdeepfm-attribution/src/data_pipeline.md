# 数据处理流程说明（与配置/代码同步）

本说明与 `agents.md` 第 4 节、`src/xdeepfm/data/preprocess.py`、YAML 配置保持一致，涵盖两类数据来源：

- **HuggingFace 数据集**（如 criteo-attribution）：`provider=huggingface`，通过 `datasets.load_dataset` 加载，缓存目录使用 `preprocess.cache_dir`（默认 `./cache/hf_datasets`）。
- **本地 CSV**（如 reczoo/criteo_X1）：`provider=local_csv`。若配置了 `dataset.split_paths.{train,valid,test}`，直接读取三份已划分好的 CSV（无需再随机或按时间切分）；否则从 `dataset.path` 读取单文件，再按 split 策略切分。

## 数据加载
1. 解析 YAML，读取 `dataset` 与 `csv_format`（分隔符、列名、header 与否）。
2. HuggingFace：调用 `load_dataset(name, split=subset, cache_dir=preprocess.cache_dir)` 转成 pandas。
3. 本地：对 `split_paths`（如 `./src/cache/local_datasets/reczoo_criteo/{train,valid,test}.csv`）逐个 `pd.read_csv`；若仅有 `path`，读入全量再进入切分逻辑。
4. 若配置了 `reader.parse_date` 等字段（针对有时间/评分列的数据），按 `reader` 规则补充派生列。

## 划分训练 / 验证 / 测试
- **已划分数据**：提供了 `split_paths` 时，直接使用 train/valid/test，不再做时间或随机切分；仍可按需要启用去重/分组一致性检查（见下）。
- **时间切分**：`split.strategy=time` 时，按 `time_column` 排序，使用 `time_split_percentiles`（如 train 70%、valid 15%、test 15%）截断；可选 `group_column` 保证同组不跨 split。
- **随机切分**：`split.strategy=random` 时，按 `train_ratio/valid_ratio/test_ratio`（seed=42）随机分配；可选 `group_column` 先对 group 抽样。
- **去重/分组一致性**：若 `dedup_column` 配置，将按优先级 `dedup_priority` 在 split 间去重；若 `group_column` 配置，确保相同 group 不落在多个 split，否则报错。
- **下采样**：支持比例与上限并存：
  - `preprocess.max_sample_ratio>0` 时，按比例抽样每个 split（ratio∈(0,1)，例如 0.03 表示保留 3%）。
  - `preprocess.max_sample>0` 时，指定每个 split 的最大行数。
  - 两者同时配置时，取二者中的较小目标行数；采样种子 `max_sample_seed`。

## 类别特征编码
1. 以 train split 计算频次（缺失填 `oov_token`），频次 < `categorical_encoding.min_freq`（默认 10）映射到 OOV。
2. 其余取值按频次排序，从 1 起编号，0 预留给 OOV/padding。
3. 生成 `vocabs/{field}.json` 和 `field -> vocab_size` 映射，写入 metadata。

## 数值特征处理
1. 按 `features.numerical` 列表对所有 split 填充缺失：`fill_na`（默认 0.0）。
2. `scaler`（默认 `standard`）在 train 上估计均值/方差，应用到所有 split，统计写入 metadata。
3. `numerical_processing.time_features.enabled` 时，为指定 `columns` 派生：
   - `delta_conversion_time = conversion_timestamp - timestamp`
   - `day_index = floor(timestamp / 86400)`
   - `hour_in_day = floor((timestamp % 86400) / 3600)`
   派生列会加入 numerical 列表并同步保存。

## 数据落地与元信息
1. 按 split 输出 Parquet：`preprocess.output_dir`（如 `./data/processed/reczoo_criteo`）下的 `train.parquet`、`valid.parquet`、`test.parquet`。
2. 写入 `metadata.yaml`：
   - `categorical_fields` 及 `vocab_sizes`
   - `numerical_fields` 与 `scaler_stats`
   - `splits`：各 split 路径与行数；若为时间切分，可记录阈值；若有去重，记录日志。
   - `extras.derived_numerical` 记录新增的时间派生列。
3. Dataloader 约定：`batch_size=4096`、`num_workers=4`、`pin_memory=true`、训练集打乱（`shuffle_train=true`），验证/测试不打乱。

## 当前专项（reczoo_criteo）的准备状态
- 数据文件：`src/cache/local_datasets/reczoo_criteo/{train,valid,test}.csv` 已存在（无需再次切分）。
- 配置：`configs/data/reczoo_criteo.yaml` 使用 `split_paths` 指向三份 CSV，`provider=local_csv`，`delimiter="\t"`，列名 I1-13/C1-26 + label。
- 采样：配置了 `max_sample_ratio=0.03`，按 3% 抽样；`max_sample=0`（不开绝对上限，可按需启用）。
- 预处理代码：`src/xdeepfm/data/preprocess.py` 已支持 `split_paths` 直接读取并跳过再划分；仍可应用去重/分组检查与标准化、编码流程。
