# xDeepFM 模型结构说明

该文档逐条对应 `agents.md` 第 5 节，为 Agent 构建 xDeepFM 提供自然语言描述。

## 5.1 输入结构
- `x_dense`：来自 YAML `features.numerical` 的标准化向量，维度 = 数值特征个数。
- `x_sparse`：`features.categorical` 的 id，维度 = 类别字段数。
- `field_vocab_sizes`：按字段顺序记录 vocab 大小，供 embedding 初始化。

## 5.2 Embedding 层
- 每个 categorical field 独立 embedding，维度 `D = model.embedding_dim = 10`。
- 生成矩阵 `X0 ∈ R^{m×D}`（m 为字段数）。
- 数值特征可直接输入线性层或并入 DNN，按 `model.dnn.input` 说明拼接。

## 5.3 FM 模块
- **一阶项**：
  - Dense 特征走线性层 (`use_first_order_dense=true`)。
  - 类别字段使用 scalar embedding (`use_first_order_categorical=true`)。
- **二阶项**：
  - 使用经典公式 `0.5 * ((∑ v_i)^2 - ∑ v_i^2)` 计算 pairwise 交互。
  - 得到标量 `fm_out`。

## 5.4 DNN 模块（隐式高阶）
- 输入为 `[x_dense, flatten(X0)]` (`model.dnn.input="concat_dense_flattened_embeddings"`)。
- 层结构：`hidden_units=[400, 400]`，每层 `relu`，`dropout=0.0`，不启用 batch norm。
- 输出向量 `dnn_out`，维度 = 400。

## 5.5 CIN 模块（显式高阶）
- 以 `X0` 为基底，`layer_sizes=[200, 200, 200]`。
- 每层执行：`X^{k-1}` 与 `X0` 做 Hadamard，经过 `layer_sizes[k]` 个卷积核（或 1x1 FC）压缩。
- 激活函数为 `identity`，不做非线性；若 `low_rank.enabled` 可进一步降维（此配置中为 false）。
- 将各层沿 embedding 维求和得到 `p^k`，拼接为 `cin_out`（维度合计 600）。

## 5.6 输出层组合
- `output_layer.combine.inputs = ["fm", "dnn", "cin"]`。
- 将 `fm_out`（标量）、`dnn_out`、`cin_out` 拼接，直接线性映射为 logit（`hidden_units=[]`，`activation="none"`）。
- 最终通过 sigmoid 得到 `ŷ ∈ (0,1)`，代表 `attribution=1` 概率。

## 5.7 其余约束
- 随机种子统一由实验配置提供 (`experiment.seed=42`)。
- 与 `configs/train/default.yaml` 对齐的损失/优化器/指标需在训练流程中引用。
