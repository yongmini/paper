
In _ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_
## 1. Introduction

- **Medical Image**의 **Classification** 및 **Semantic Segmentation** 작업에서 흔히 발생하는 **Class Imbalance** 문제는 **Deep Learning Models**의 성능을 저해하는 주요 요인 중 하나

- **Classification Task**에서는 실제 임상에서의 다양한 질병 발생률이 **Medical Image Datasets**의 클래스 분포에 불균형을 초래했음

- **Semantic Segmentation Task**에서는, foreground  클래스 픽셀이 작은 영역을 차지하는 반면, background 클래스 픽셀이 대다수를 차지하여 심한 불균형이 발생했음. 데이터셋에 **Class Imbalance**가 있을 경우, **Head Classes**는 훈련 과정에서 지배적인 역할을 하였으며, 이는 **Tail Classes**의 성능 저하를 초래하고, **Deep Learning Models**의 **Diagnostic Effectiveness**에 심각한 영향을 미쳤음

- **Class Imbalance** 문제를 해결하는 일반적인 방법으로는 **Training Data Re-sampling** 또는 **Loss Function Re-weighting**이 있었음. 최근에는 **Dynamic Curriculum Learning**, **Multi-Experts Ensemble Learning**, **Two-Stage Learning** 등 다양한 **Re-Balancing Methods**가 제안되었음

- 이 중 **Two-Stage Learning**은 가장 우수한 성과를 거두었음. **Two-Stage Learning**은 모델의 학습 과정을 **Representation Learning**과 **Classifier Learning**의 두 단계로 나누었음

- **Representation Learning** 단계는 매우 중요했음. **High-Quality Representations**는 **Downstream Tasks**의 정확도에 직접적인 영향을 미쳤기 때문임. 기존 연구들은 **Cross-Entropy Loss**보다 **Supervised Contrastive Learning (SCL)**이 이상적인 **Feature Space Representation** 학습에 더 유리하다는 것을 보여주었음

- 문제는 데이터셋에 **Class Imbalance**가 있을 경우, **SCL**은 **Head Classes**를 최적화하는 데 더 많은 노력을 기울이게 되어, 클래스들이 균일하게 분포하지 않게 되고, **Tail Classes**는 **Feature Confusion**에 취약해짐 실제로 **Medical Images**는 **Natural Images**보다 클래스 간 **Appearance Difference**가 더 작음 이러한 이유로, 의료 이미지에서 **Tail Classes** 간 **Feature Confusion**은 더 심각하게 발생할 가능성이 있음

- 이 문제를 해결하기 위해,  **Balanced**하고 **Discriminative Representation**을 얻기 위한 새로운 **Balanced and Discriminative Contrastive Learning (BDCL)** 방법을 제안하였음

- **BDCL**은 **SCL**을 기반으로 두 가지 개선 사항을 도입하였음

- 첫째, **Gradient Analysis**를 통해 **SCL**이 **Hardness-Aware Loss Function**임을 보여주었으며, **Temperature (τ)**가 **Hard Negative Samples**에 대한 패널티 강도를 제어할 수 있음을 발견하였음. 각 클래스에 **Temperature**를 설정하고, 훈련 과정에서 **Features** 간의 유사성에 따라 클래스별 **Temperature**를 동적으로 조정하여, 다른 클래스에서 온 **Negative Samples**의 **Gradient Contribution**을 균형 있게 맞추었음

- 둘째, 각 클래스의 **Hard Example Prototype**을 비교 인스턴스로 **Contrastive Loss**에 도입하여, 모든 클래스가 매 미니배치에서 **Contrastive Loss** 계산에 포함되도록 하였고, 이는 **Tail Classes** 간의 세부적인 **Feature Differences**를 더 잘 학습할 수 있도록 하였음

- 다양한 불균형한 **Medical Image Datasets**에 대한 실험 결과, **BDCL**은 네트워크 모델이 **Balanced**하고 **Discriminative Representations**를 학습하도록 하여 **Class Imbalance** 문제를 효과적으로 해결하였음

## 2. Proposed Method

- 본 논문에서는 **Two-Stage Learning Framework**를 사용하였음. **Representation Learning** 단계에서는 **BDCL**을 사용하여 **Backbone Network**를 훈련시켜 **Balanced**하고 **Discriminative Representation**을 얻었음 **Classifier Learning** 단계에서는 **Backbone Network**를 고정한 후, **Class-Balanced Sampling**을 통해 **Classifier**를 미세 조정하였음

### 2.1. Preliminaries

- **Cross-Entropy Loss**와 비교할 때, **Supervised Contrastive Learning (SCL)**은  **Feature Space**에서 **Feature Distribution**의 균일성을 더욱 촉진하여, **Positive Sample Features**를 가깝게 모으고 **Negative Sample Features**를 멀리 밀어내는 데 더 유리하였음. **SCL**의 표현식은 다음과 같음:

$$L_{SCL}(x_i) = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log\left(\frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}\right)$$

- 여기서, $z_i$는 $x_i$의 **Feature**를 나타내고, $A(i)$는 $z_i$를 제외한 현재 배치의 **Feature Set**을 나타냄. $P(i)$는 $x_i$와 동일한 **Label**을 가진 **Positive Sample**들의 집합임. $\tau$는 **Temperature Parameter**임

### 2.2. Temperature Dynamic Learning

- **SCL**에서 **Temperature**가 **Negative Samples**에 대한 패널티 강도를 제어한다는 것을 **Gradient Analysis**를 통해 보여주었음 **Anchor Sample**인 $x_i$와 **Negative Sample**인 $z_p^-$의 유사성에 대한 **Gradient**는 다음과 같음:

$$\frac{\partial L_{SCL}(x_i)}{\partial z_i \cdot z_p^-} = \frac{1}{\tau} \frac{\exp(z_i \cdot z_p^- / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}$$

여기서, $P^-(i)$는 $x_i$와 다른 **Label**을 가진 **Negative Samples**들의 집합을 나타냄 **Negative Sample**의 상대적 패널티를 나타내기 위해 $r_i(z_i \cdot z_p^-)$를 정의하였음.

훈련 과정에서 각 클래스의 **Temperature**를 동적으로 학습하였고, **SCL**의 **Gradient**는 다음과 같음:

$$\frac{\partial L_{SCL}(x_i)}{\partial \tau'_{y_i}} = \frac{1}{|P(i)|} \sum_{p \in P(i)} \frac{(1 - s_{i,p})}{(\tau'_{y_i})^2}$$

이 **Gradient**를 사용하여 클래스별 **Temperature**를 업데이트하였음**Negative Sample**이 **Anchor Sample**과 높은 유사성을 가질 경우, 해당 클래스의 **Temperature**를 줄여 **Negative Sample**에 대한 패널티 강도를 높였음

### 2.3. Hard Example Prototypes

- **Class Imbalance**로 인해 **Tail Classes**가 각 미니배치에 항상 포함되지 않을 수 있음. 이를 해결하기 위해 우리는 **Prototypes**를 도입하였음. 모든 샘플의 **Feature Mean** 대신, 각 클래스의 **Hard Example**들을 찾아 해당 **Feature Mean**을 **Prototype**으로 사용하였음 (음)

- 각 미니배치에 포함된 하드 예시만을 사용하여 **Prototype**을 생성하기 위해 **Exponential Moving Average**를 사용하였음:

$$
z^*_c = (1 - m) \cdot z^*_c + m \cdot z^t_{c, c \neq \bar{c}}
$$

- 여기서 $z^*_c$는 클래스 $c$의 **Hard Example Prototype**이고, $m$은 **Momentum Coefficient**임

- 최종적으로 **BDCL Loss**는 다음과 같이 정의되었음

$$
L_{BDCL}(x_i) = -\frac{1}{|P(i)| + 1} \times \sum_{p \in \{P(i)\} \cup \{c\}} \log\left( \frac{\exp(z_i \cdot z_p / \tau'_{y_i})}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau'_{y_i}) + \sum_{k \in C} \exp(z_i \cdot z^*_k / \tau'_{y_i})} \right)
$$

- **Classification Task**에서는 **Inter-Image Comparisons**를 수행하며, 같은 **Label**을 가진 이미지를 **Positive Samples**로, 다른 **Label**을 가진 이미지를 **Negative Samples**로 취급하였음

- **Semantic Segmentation Task**에서는 **Pixel-to-Pixel Comparisons**를 수행하였. 각 이미지의 픽셀 중 같은 **Class**에 속하는 다른 픽셀들을 **Positive Samples**로, 다른 **Class**에 속하는 픽셀들을 **Negative Samples**로 설정하였음.
## 3. Experiments

### 3.1. Dataset

- 분류 작업에서, **BDCL**을 두 가지 공개적으로 이용 가능한 클래스 불균형 의료 이미지 데이터셋인 **ISIC2019**와 **Hyper-Kvasir**에서 평가하였음  **ISIC2019** 데이터셋은 피부 병변 진단 작업을 위해 구성되었으며, 8개의 클래스와 25,331개의 피부경 검사를 포함하고 있음 **Hyper-Kvasir** 데이터셋은 23개의 클래스와 10,662개의 내시경 이미지로 구성되어 있으며 특히 불균형도가 최대 184에 달하는 매우 불균형한 데이터를 제공함

- Semantic Segmentation에서,  **Tianjin Medical University Cancer Institute and Hospital**의 **Thyroid Ultrasound Image (TUI)** 데이터셋을 사용하여 **BDCL**을 평가하였음 **TUI**는 10,193개의 이미지와 3개의 의미적 클래스(배경, 양성 종양, 악성 종양)에 대한 정밀한 주석을 포함하고 있음 배경 클래스 픽셀은 전체 픽셀의 92%를 차지하며, 양성 종양과 악성 종양 픽셀은 각각 5%와 3%를 차지함

- 이전 연구를 따르며, 각 데이터셋을 8:2 비율로 훈련 세트와 테스트 세트로 나누었음

### 3.2. Image Classification Experiments

- 분류 작업에서,  **Accuracy (ACC)**, **G-mean**, **F1-Score**, **Weighted Index of Balanced Accuracy (IBAα)** 등의 네 가지 평가 지표를 선택하였음 공정성을 위해 **ResNet-50**을 **Backbone**으로 사용하였음.  초기 학습률 0.01, 모멘텀 0.9, 가중치 감쇠 0.0001을 설정한 **SGD**를 **Optimizer**로 사용하였음

- **BDCL**은 모든 평가 지표에서 최고의 성능을 달성하였음. **T-SNE** 특징 시각화 했는데 **SCL**로 훈련된 네트워크 모델이 클래스 간의 심각한 **Feature Confusion**을 겪고 있음을 확인할 수 있었음.반면, **BDCL**로 훈련된 네트워크 모델이 추출한 **Features**는 더 균일하게 분포하고, 클래스 간 선형 분리가 가능했음. 이는 **BDCL**이 클래스 내 밀집도와 클래스 간 분산을 더 잘 학습할 수 있음을 입증함.

| Methods        | ISIC2019 |        |          |        |        |       | Hyper-Kvasir |        |          |        |        |       |
| -------------- | -------- | ------ | -------- | ------ | ------ | ----- | ------------ | ------ | -------- | ------ | ------ | ----- |
|                | ACC      | G-mean | F1-score | IBA0.1 | IBA0.5 | IBA1  | ACC          | G-mean | F1-score | IBA0.1 | IBA0.5 | IBA1  |
| CE             | 79.12    | 78.49  | 69.10    | 60.17  | 53.15  | 44.36 | 86.66        | 60.59  | 55.29    | 54.12  | 51.27  | 47.71 |
| Focal          | 80.44    | 80.26  | 71.45    | 63.06  | 56.30  | 47.86 | 86.80        | 61.42  | 55.16    | 53.82  | 51.10  | 47.69 |
| LDAM           | 82.40    | 82.25  | 74.52    | 66.31  | 59.77  | 51.59 | 89.82        | 64.17  | 59.24    | 57.45  | 54.93  | 51.79 |
| cRT            | 81.06    | 84.16  | 72.99    | 69.56  | 63.48  | 55.89 | 90.06        | 65.74  | 59.34    | 58.46  | 55.94  | 52.78 |
| TSC            | 82.75    | 84.26  | 76.27    | 69.78  | 63.61  | 55.89 | 90.39        | 64.01  | 58.95    | 57.59  | 55.27  | 52.35 |
| Bal-Mxp        | 81.55    | 82.27  | 73.86    | 66.37  | 59.86  | 51.72 | 90.48        | 64.74  | 59.66    | 58.39  | 55.88  | 52.74 |
| GCL            | 82.60    | 81.64  | 74.55    | 65.53  | 59.02  | 50.89 | 89.77        | 66.01  | 59.14    | 57.78  | 55.26  | 52.09 |
| **BDCL(ours)** | 84.57    | 84.96  | 76.59    | 70.92  | 64.77  | 57.10 | 91.33        | 66.05  | 60.50    | 59.25  | 56.95  | 54.07 |
### 3.3. Semantic Segmentation Experiments

- 의미론적 분할 작업에서는 각 클래스의 **IOU (Intersection over Union)**와 평균 **IOU (mIOU)**를 보고하였음 공정성을 위해 **HRNetV2-W48**을 **Backbone Network**로, **OCR**을 **Semantic Segmentation Head**로 사용하였음 초기 학습률 0.01, 모멘텀 0.9, 가중치 감쇠 0.0005를 설정한 **SGD**를 **Optimizer**로 사용하였음

| Methods    | Background | Benign | Malignant | mIOU |
|------------|------------|--------|-----------|------|
| CE         | 98.53      | 79.80  | 74.76     | 84.36|
| Lovasz     | 98.71      | 81.31  | 73.16     | 84.39|
| Focal      | 98.73      | 80.92  | 75.02     | 84.89|
| LDAM       | 98.83      | 82.22  | 77.08     | 86.05|
| SoftIou    | 98.84      | 82.05  | 75.67     | 85.52|
| TSC        | 98.83      | 82.32  | 76.89     | 86.01|
| **BDCL(ours)** | 98.86      | 83.39  | 78.42     | 86.89|

### 3.4. Ablation Experiments

- **BDCL**의 다양한 구성 요소의 효과를 검증하기 위해, 우리는 **Temperature Dynamic Learning (TDL)** 또는 **Hard Example Prototypes (HEP)**을 제거한 후 **Ablation Experiments**를 수행하였음  **TDL** 또는 **HEP** 중 하나만을 사용하는 경우 성능 향상이 미미하였음 반면, 두 가지 구성 요소를 모두 적용했을 때 성능이 크게 향상되었으며, 이는 두 구성 요소가 클래스 불균형 문제 해결에 필수적임을 나타냄

-  **TDL**이 클래스별 **Temperature**를 동적으로 학습할 수 있음을 검증하기 위해 각 클래스의 **Temperature Parameter** 변화를 시각화하였음  **Head Classes**는 훈련 중에 더 높은 **Temperature**를 가졌고, **Tail Classes**는 더 낮은 **Temperature**를 가졌음

| Methods    | Accuracy | G-mean | F1-score | IBA0.1 | IBA0.5 | IBA1 |
|------------|----------|--------|----------|--------|--------|------|
| baseline   | 82.06    | 84.02  | 74.00    | 69.20  | 62.80  | 54.79|
| w/o TDL    | 83.01    | 84.26  | 73.87    | 69.63  | 63.28  | 55.34|
| w/o HEP    | 83.72    | 84.57  | 74.55    | 70.20  | 63.99  | 56.24|
| **BDCL**   | 84.57    | 84.96  | 76.59    | 70.92  | 64.77  | 57.10|

## 4. Conclusion

- 본 논문에서는 클래스 불균형이 있는 **Medical Images**에서의 **Representation Learning** 단계에 초점을 맞추어, 새로운 **Balanced and Discriminative Contrastive Learning (BDCL)** 방법을 제안하였음 

- **BDCL**은 두 가지 구성 요소로 이루어져 있음: **Temperature Dynamic Learning (TDL)**과 **Hard Example Prototypes (HEP)**. **TDL**은 훈련 중에 클래스별 **Temperature**를 동적으로 학습하여, 서로 다른 클래스에서 온 **Negative Samples**의 **Gradient Contribution**을 균형 있게 맞추었음

- **HEP**는 **Hard Example Prototypes**을 **Contrastive Loss**에 도입하여, **Tail Classes** 간의 특징을 더 잘 구별할 수 있게 하였음 실험 결과, **BDCL**은 네트워크 모델이 **Balanced**하고 **Discriminative Representations**를 학습하게 만들어, 클래스 불균형 문제를 효과적으로 해결하였음을 입증하였음
