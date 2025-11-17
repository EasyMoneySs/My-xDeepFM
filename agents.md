# xDeepFM 复现完整方案（基于 `criteo/criteo-attribution-dataset` + YAML 配置）

> 目标：
> 用 **HuggingFace 上的 `criteo/criteo-attribution-dataset`**，配置全部用 **YAML** 管理，给“AI 开发助手”（比如 Codex/agent）一套 **从数据到训练到评估** 的完整可执行方案（不写具体代码，只给清晰结构 + 配置 + 步骤）。

---

## 1. 项目整体结构设计

建议项目目录结构如下（AI 工具照着建即可）：

```text
xdeepfm-attribution/
├── configs/
│   ├── data/
│   │   └── criteo_attribution.yaml      # 数据相关配置
│   ├── model/
│   │   └── xdeepfm.yaml                 # 模型结构 & 超参配置
│   ├── train/
│   │   └── default.yaml                 # 训练 & 优化器 & 评估配置
│   └── experiment/
│       └── xdeepfm_criteo_attr.yaml     # 组合 data/model/train 的实验配置
├── src/
│   ├── data_pipeline.md                 # 数据处理逻辑说明（自然语言）
│   ├── model_spec.md                    # 模型结构说明（自然语言）
│   ├── training_flow.md                 # 训练流程说明（自然语言）
│   └── entrypoints/
│       ├── run_preprocess.md            # 预处理入口说明
│       ├── run_train.md                 # 训练入口说明
│       └── run_eval.md                  # 评估入口说明
└── README.md                            # 顶层说明（可引用本方案）
```

> 说明：
>
> * 你可以让 AI 根据这些 Markdown 说明，自动生成对应的 `.py` 代码文件；
> * 这里我们只给 **结构 + 配置 + 过程描述**，不直接写代码。

---

## 2. 数据与任务定义

使用的数据集：**HuggingFace `criteo/criteo-attribution-dataset`**

典型字段（简化）：

* `timestamp`：展示发生时间（秒，0 开始，递增）
* `uid`：用户 ID（类别型）
* `campaign`：广告活动 ID（类别型）
* `click`：是否点击（0/1）
* `conversion`：展示后 30 天内是否有转化（0/1）
* `attribution`：该展示是否被归因为这次转化的“负责展示”（0/1）
* `conversion_timestamp`：转化发生时间（秒，或 -1）
* `conversion_id`：转化 ID（或 -1）
* `click_pos`：点击在该转化路径中的位置
* `click_nb`：该转化路径总点击次数
* `cost`：展示成本（已变换）
* `cpo`：有归因转化时的每转化成本（已变换）
* `time_since_last_click`：距上一次点击的时间（秒）
* `cat1`…`cat9`：匿名上下文类别特征（类别型）

### 2.1 推荐任务：Attribution Prediction（二分类）

* **输入特征**：用户、广告、上下文、行为特征
* **标签 label**：`attribution`（0 / 1）
* **目标**：预测每一次展示被归因的概率（类似“这次展示对转化负责的概率”）

### 2.2 特征划分建议

在 YAML 里显式写清楚哪些是 categorical / numerical：

* **categorical_features**：

  * `uid`
  * `campaign`
  * `cat1`…`cat9`

* **numerical_features**：

  * `timestamp`（可转成相对时间，如天 / 小时）
  * `click`
  * `conversion`
  * `click_pos`
  * `click_nb`
  * `cost`
  * `cpo`
  * `time_since_last_click`
  * `conversion_timestamp`（可转换为 “距展示的时间差”）

* **label_column**：

  * `attribution`

---

## 3. YAML 配置设计

### 3.1 数据配置：`configs/data/criteo_attribution.yaml`

```yaml
dataset:
  provider: "huggingface"
  name: "criteo/criteo-attribution-dataset"
  # 子集名称（如果有 train/validation/test 子集，可以指定）
  subset: null

task:
  type: "binary_classification"
  label_column: "attribution"       # 可切换为 "click" 或 "conversion"

features:
  categorical:
    - "uid"
    - "campaign"
    - "cat1"
    - "cat2"
    - "cat3"
    - "cat4"
    - "cat5"
    - "cat6"
    - "cat7"
    - "cat8"
    - "cat9"
  numerical:
    - "timestamp"
    - "click"
    - "conversion"
    - "click_pos"
    - "click_nb"
    - "cost"
    - "cpo"
    - "time_since_last_click"
    - "conversion_timestamp"

preprocess:
  cache_dir: "./cache/hf_datasets"
  output_dir: "./data/processed/criteo_attribution"

  # 是否基于时间切分（更贴近线上场景）
  split:
    strategy: "time"    # "time" 或 "random"
    time_column: "timestamp"
    # 如果是 time 策略，可以按百分位数切分
    time_split_percentiles:
      train: 0.7        # 前 70% 时间段
      valid: 0.15       # 接下来的 15%
      test: 0.15        # 最后 15%

    random:
      train_ratio: 0.8
      valid_ratio: 0.1
      test_ratio: 0.1
      seed: 42

  categorical_encoding:
    method: "index"     # 将每个字段的取值映射到 0..N 的整数 id
    min_freq: 10        # 低频类别合并为 OOV
    oov_token: "__OOV__"

  numerical_processing:
    fill_na: 0.0
    scaler: "standard"  # "standard", "minmax", or "none"
    # 如需将时间戳转成“相对秒/小时/天”等特征，可由代码按照此策略实现
    time_features:
      enabled: true
      columns:
        - "timestamp"
        - "conversion_timestamp"
      # 例如生成：相对时间 / 天内小时等
      derived_features:
        - "delta_conversion_time"   # = conversion_timestamp - timestamp（若无转化则设为 0 或 -1）
        - "day_index"               # = timestamp / (24*3600) 取整
        - "hour_in_day"             # = (timestamp % (24*3600)) / 3600

dataloader:
  batch_size: 4096
  num_workers: 4
  pin_memory: true
  shuffle_train: true
```

---

### 3.2 模型配置：`configs/model/xdeepfm.yaml`

```yaml
model:
  name: "xDeepFM"

  # Embedding
  embedding_dim: 10   # 与论文保持一致

  # FM 部分（线性 + 二阶）
  fm:
    use_fm: true
    # 是否包含 dense 特征的一阶项
    use_first_order_dense: true
    use_first_order_categorical: true

  # DNN 部分（隐式高阶 bit-wise 交互）
  dnn:
    use_dnn: true
    input: "concat_dense_flattened_embeddings"
    hidden_units: [400, 400]
    activation: "relu"
    dropout: 0.0
    batch_norm: false

  # CIN 部分（显式高阶 vector-wise 交互）
  cin:
    use_cin: true
    layer_sizes: [200, 200, 200]   # Criteo 上推荐配置
    activation: "identity"         # 论文中实验显示不加非线性效果更好
    # 是否使用低秩分解（可选，控制参数量）
    low_rank:
      enabled: false
      rank: 16

  # 输出层
  output_layer:
    # 将 FM 输出（标量）、DNN 输出（向量）、CIN 输出（向量）拼接后
    # 通过一层全连接映射为单一 logit
    combine:
      inputs: ["fm", "dnn", "cin"]
      hidden_units: []             # 空列表表示直接 Linear -> 1
      activation: "none"
```

---

### 3.3 训练配置：`configs/train/default.yaml`

```yaml
training:
  device: "cuda"            # 或 "auto": 自动检测 cuda / cpu
  epochs: 20
  early_stopping:
    enabled: true
    metric: "valid_logloss"
    mode: "min"
    patience: 2
    delta: 0.0001

  optimizer:
    type: "adam"
    lr: 0.001
    weight_decay: 0.0001   # L2 正则

  lr_scheduler:
    type: "none"           # 或 "step", "cosine" 等
    step_size: 5
    gamma: 0.5

  loss:
    type: "binary_cross_entropy"

  metrics:
    - "logloss"
    - "auc"

  checkpoint:
    save_dir: "./checkpoints/xdeepfm_criteo_attr"
    monitor: "valid_logloss"
    mode: "min"
    save_best_only: true
```

---

### 3.4 实验组合配置：`configs/experiment/xdeepfm_criteo_attr.yaml`

可以把 data / model / training 三块组合起来，方便一行命令跑完整实验：

```yaml
experiment:
  name: "xdeepfm_criteo_attribution"

  # 引用其它配置文件对应的 node
  data_config: "configs/data/criteo_attribution.yaml"
  model_config: "configs/model/xdeepfm.yaml"
  train_config: "configs/train/default.yaml"

  # 运行输出路径（日志、结果等）
  output_dir: "./runs/xdeepfm_criteo_attr"

  # 随机种子统一配置
  seed: 42
```

> 说明：
>
> * 具体“如何 include 其它 YAML”可以让 AI 根据你使用的框架（Hydra / 自定义解析器）来自动生成；
> * 关键是 **信息都在 YAML 里**，AI 很容易用这些配置生成对应代码。

---

## 4. 数据处理流程（给 AI 的自然语言说明）

放在 `src/data_pipeline.md`，大致可以这样写（示意）：

### 4.1 数据加载

1. 使用 `datasets.load_dataset("criteo/criteo-attribution-dataset")` 从 HuggingFace 加载数据；
2. 根据 `configs/data/criteo_attribution.yaml`：

   * 读取 `features.categorical` / `features.numerical` / `task.label_column`；
   * 指定缓存目录 `preprocess.cache_dir`。

### 4.2 划分训练 / 验证 / 测试

1. 如果 `preprocess.split.strategy == "time"`：

   * 按 `preprocess.split.time_column`（默认 `timestamp`）对全体样本排序；
   * 根据 `time_split_percentiles` 划分：

     * 0% ~ 70% → train
     * 70% ~ 85% → valid
     * 85% ~ 100% → test
2. 如果 `strategy == "random"`：

   * 使用 `train_ratio / valid_ratio / test_ratio` + `seed` 做随机划分。

### 4.3 类别特征编码

对于 `features.categorical` 中的每一列：

1. 统计该特征所有取值的频次；
2. 对频次 < `preprocess.categorical_encoding.min_freq` 的取值统一映射到 `oov_token`；
3. 为每个剩余取值分配一个 id，从 1 开始；id=0 预留给 padding / OOV；
4. 保存每个特征的 `vocab_size`，用于构建 embedding。

输出：

* `train / valid / test` 中该列都变成 int id；
* 一个字典 `field_name → vocab_size`。

### 4.4 数值特征处理

对于 `features.numerical` 中的每一列：

1. 按 `preprocess.numerical_processing.fill_na` 填充缺失；
2. 如果 `scaler == "standard"`：

   * 在训练集上计算均值 / 标准差；
   * 使用相同参数对 train/valid/test 做标准化；
3. 如果存在 `time_features.enabled == true`：

   * 根据 `time_features.derived_features` 创建新特征，例如：

     * `delta_conversion_time = max(conversion_timestamp - timestamp, 0)`，若没有转化，可填 0 或 -1；
     * `day_index = floor(timestamp / (86400))`；
     * `hour_in_day = floor((timestamp % 86400) / 3600)`；
   * 将这些新特征也视作 numerical_features 处理。

### 4.5 数据落地格式

* 建议将处理后的 train/valid/test 分别保存为：

  * 二进制格式（如 parquet / arrow / numpy / pkl 任意你喜欢）；
  * 同时保存一个元信息文件（如 JSON/YAML）记录：

    * 数值列名；
    * 类别列名；
    * 每个类别列的 vocab_size；
    * 数值列的标准化参数（均值 / 方差）。

AI 在训练时，只需要读取这些信息即可构造 DataLoader。

---

## 5. 模型结构说明（给 AI 的自然语言说明）

放在 `src/model_spec.md`，重点描述 **xDeepFM 三个模块 + 输出层**：

### 5.1 输入结构

* 对每个样本：

  * `x_dense`: 所有 numerical_features 构成的向量，维度 = `num_numerical_features`;
  * `x_sparse`: 所有 categorical_features 的 id，维度 = `num_categorical_features`;
* 模型还需要一个元信息：

  * `field_vocab_sizes`: 一个 list，长度 = num_categorical_features，记录每个 field 的 vocab_size。

### 5.2 Embedding 层

* 对 `x_sparse` 的每个 field i：

  * 使用一个维度为 `model.embedding_dim` 的 embedding 矩阵；
  * 得到 embedding 向量 `e_i ∈ R^D`；
* 所有字段的 embedding 拼成矩阵：

  * `X0 ∈ R^{m × D}`，其中 m = `num_categorical_features`。

### 5.3 FM 模块

* **一阶项**：

  * 对 dense 特征使用 Linear；
  * 对每个类别字段使用一个 scalar embedding。
* **二阶项**：

  * 使用经典 FM 计算公式，对所有 field 的 embedding 进行二阶交互；
  * 输出为一个标量 `fm_out`。

### 5.4 DNN 模块（隐式高阶）

* 输入 = `[x_dense, flatten(X0)]`，维度 = `num_numerical + m * D`；
* 结构：

  * 多层全连接，层数 = `len(model.dnn.hidden_units)`；
  * 每层神经元个数按 `hidden_units` 列表；
  * 激活函数 = `model.dnn.activation`（例如 ReLU）；
  * 按 `model.dnn.dropout` 配置添加 Dropout；
* 输出为向量 `dnn_out`，维度 = 最后一层隐藏单元数。

### 5.5 CIN 模块（显式高阶）

* 输入 = `X0 ∈ R^{m × D}`；
* 对每一层 k：

  * 上一层输出 `X^{k-1} ∈ R^{H_{k-1} × D}`;
  * 与 `X0` 做逐维 Hadamard 交互，产生形状 `(H_{k-1}, m, D)` 的张量；
  * 在 `(H_{k-1}, m)` 上通过 `layer_sizes[k]` 个线性滤波器压缩，得到新层的 `X^k ∈ R^{H_k × D}`;
  * 对每个 `X^k` 在维度 D 上求和，得到 `p^k ∈ R^{H_k}`;
* 将所有 `p^k` 拼接，得到 `p_plus ∈ R^{sum(layer_sizes)}`，即 `cin_out`。

### 5.6 输出层组合

* 将三个分支的输出拼接：

  * `concat = [fm_out, dnn_out, cin_out]`;
* 经过输出层：

  * 如果 `output_layer.hidden_units` 为空，则直接 `Linear -> sigmoid`；
  * 否则可以多一到两层 MLP 再 sigmoid；
* 最终输出为一个标量 `ŷ ∈ (0,1)`，表示 `attribution=1` 的概率。

---

## 6. 训练流程说明（给 AI 的自然语言说明）

放在 `src/training_flow.md`。

### 6.1 训练入口 `run_train` 的逻辑

1. 读取 `configs/experiment/xdeepfm_criteo_attr.yaml`；
2. 根据其中路径加载：

   * 数据配置 YAML；
   * 模型配置 YAML；
   * 训练配置 YAML；
3. 使用数据配置：

   * 若 processed 数据不存在，则先执行预处理流程；
   * 构建 Train/Valid/Test DataLoader；
4. 使用模型配置：

   * 构建 XDeepFM 模型实例；
   * 初始化 embedding 等参数；
5. 使用训练配置：

   * 设置 optimizer, lr_scheduler, loss, metrics；
   * 进行多轮 epoch 训练；
   * 每个 epoch：

     * 训练阶段：遍历 train_loader，累积 loss；
     * 验证阶段：遍历 valid_loader，计算 logloss & AUC；
     * 根据 early_stopping 配置，决定是否保存 checkpoint / 提前终止；
6. 训练结束后：

   * 加载最佳 checkpoint；
   * 在 test_loader 上评估 logloss & AUC；
   * 将结果保存到 `experiment.output_dir` 下的一个 JSON / YAML 文件。

### 6.2 评估指标定义

* **Logloss**：二分类交叉熵；
* **AUC**：ROC AUC；
* 可以附加统计：

  * 正负样本比例；
  * 不同 bucket（例如 cost 区间、time_since_last_click 区间）下的 AUC。

---

## 7. 最终执行顺序总结（给 AI 的操作 checklist）

1. **预处理阶段**

   * 读取 `configs/data/criteo_attribution.yaml`；
   * 从 HuggingFace 加载 `criteo/criteo-attribution-dataset`；
   * 按时间或随机方式划分 train/valid/test；
   * 对 categorical 特征做频次过滤 + id 映射；
   * 对 numerical 特征做填充 + 标准化 + 派生时间特征（如需要）；
   * 保存处理后的数据，以及：

     * vocab_sizes；
     * 标准化参数；
     * 划分方式记录。

2. **模型构建阶段**

   * 读取 `configs/model/xdeepfm.yaml`；
   * 根据数据元信息构建 Embedding 层（每个 category field 一个 embedding 矩阵）；
   * 按配置搭建 FM、DNN、CIN 三个分支；
   * 搭建输出层，将三路输出组合。

3. **训练阶段**

   * 读取 `configs/train/default.yaml`；
   * 构建 optimizer（Adam）、loss（BCE）、metrics（logloss & AUC）；
   * 按 epoch 训练，记录训练/验证指标；
   * 根据 early_stopping 保存最优模型。

4. **测试与结果保存**

   * 在 test 集上计算 logloss & AUC；
   * 将最终结果写入 JSON/YAML，如：

     * `runs/xdeepfm_criteo_attr/metrics_test.yaml`；
   * 记录主要超参、数据划分方式，以便复现。

---
