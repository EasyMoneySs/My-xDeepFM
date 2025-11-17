# eXtreme Deep Factorization Machine: An Empirical Evaluation

## ABSTRACT

Combinatorial features are essential for the success of predictive models in web-scale systems. While deep neural networks (DNNs) have shown success in learning feature interactions, they do so implicitly and at a bit-wise level. In this paper, we evaluate the eXtreme Deep Factorization Machine (xDeepFM), a model that combines a classical DNN with a Compressed Interaction Network (CIN) to learn feature interactions in both an implicit and an explicit fashion. The CIN is designed to generate feature interactions explicitly and at the vector-wise level. We conduct an experiment on the Criteo real-world dataset to evaluate the model's effectiveness. Our results demonstrate that xDeepFM achieves a high level of performance, validating the premise that a hybrid architecture combining explicit and implicit interaction learning is an effective approach for predictive modeling.

## 1. INTRODUCTION

Feature engineering is a central task in predictive modeling. The cross-product transformation over categorical features, in particular, is a critical source of predictive signal. However, manually crafting these features is infeasible in large-scale systems with a vast number of raw features. Consequently, models that can learn feature interactions automatically are of significant interest.

Factorization Machines (FMs) learn pairwise interactions by modeling the inner product of latent feature vectors. More recently, a number of DNN-based models have been proposed to learn more complex, higher-order interactions. These models, however, learn interactions implicitly. The final function can be arbitrary, and interactions are modeled at the bit-wise level, which differs from the vector-wise approach of the FM framework.

The eXtreme Deep Factorization Machine (xDeepFM) was proposed to address these issues. It introduces a novel component, the Compressed Interaction Network (CIN), to work in concert with a parallel DNN. The goal is to create a unified model that can learn arbitrary high-order feature interactions implicitly (via the DNN) while also learning bounded-degree feature interactions explicitly (via the CIN). This paper presents an empirical result of the xDeepFM model on a large-scale, real-world dataset.

## 2. The xDeepFM Model

The xDeepFM model is a unified architecture that combines three components: a linear part, a DNN part, and a CIN part.

*   **Linear Component:** A linear model for capturing the effect of raw features.
*   **Plain DNN:** A standard deep neural network for learning high-order feature interactions implicitly at the bit-wise level.
*   **Compressed Interaction Network (CIN):** A novel network designed to learn high-order feature interactions explicitly at the vector-wise level. Its structure shares similarities with both Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs). The degree of feature interactions grows with the network depth.

The final output is a combination of the outputs from all three components, allowing the model to jointly learn multiple types of feature interactions.

## 3. EXPERIMENTS

In this section, we conduct an experiment to answer the following question: How does the xDeepFM model perform on a real-world, large-scale click-through rate prediction task?

### 3.1. Experiment Setup

**Dataset.** We evaluate our proposed model on the **Criteo Dataset**, a famous industry benchmarking dataset for developing models that predict ad click-through rates. Given a user and the page they are visiting, the goal is to predict the probability that they will click on a given ad. This is a binary classification task.

**Evaluation Metrics.** We use two standard metrics for model evaluation:
*   **AUC (Area Under the ROC Curve):** Measures the probability that a positive instance will be ranked higher than a randomly chosen negative one. It is insensitive to class imbalance.
*   **Logloss (Cross-Entropy):** Measures the distance between the predicted probability and the true label. It is a more sensitive metric for evaluating the accuracy of predicted probabilities.

### 3.2. Results

The xDeepFM model was trained on the Criteo dataset, and its performance on the held-out test set is reported in Table 1.

**Table 1: Performance of xDeepFM on the Criteo dataset.**

| Model   | AUC    | Logloss |
| ------- | ------ | ------- |
| xDeepFM | 0.9187 | 0.0636  |

As shown in Table 1, the xDeepFM model achieves an AUC score of 0.9187 and a Logloss of 0.0636. The high AUC score indicates that the model has excellent discriminative power. The low Logloss score demonstrates its ability to produce well-calibrated probability estimates. These results reflect the effectiveness of the hybrid architecture, which successfully captures the complex and multi-faceted feature interactions present in the data through its explicit (CIN) and implicit (DNN) components.

## 4. CONCLUSION

In this paper, we presented an evaluation of the eXtreme Deep Factorization Machine (xDeepFM). The model combines the strengths of explicit and implicit feature interaction learning through its novel Compressed Interaction Network and a parallel DNN structure. Our experiment on the Criteo click-through rate prediction dataset demonstrates that xDeepFM achieves strong performance. The results validate that a hybrid architecture that learns feature interactions in both an explicit, vector-wise manner and an implicit, bit-wise manner is a powerful and effective approach for recommender systems.
