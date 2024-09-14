29th Conference on Neural Information Processing Systems (NIPS 2016)

## 1. Introduction

- 시뮬레이션 데이터로 학습한 모델을 실제 도메인에 일반화하기 어려운 문제를 해결하기 위해, 레이블 없는 target domain에 knowledge transfer를 목표로 도메인 불변 표현을 학습하겠음

- Object Classification과 Pose Estimation 작업에서 Source와 Target Domain 모두에서 관심 Object가 주어진 이미지의 Foreground에 위치함

- Source와 Target Domain의 픽셀 분포는 여러 방식으로 다를 수 있는데,  이러한 차이를 "Low-Level"과 "High-Level"로 나누었음 

- "Low-Level" 차이는 노이즈, 해상도, 조명, 색상과 같은 요소로 인한 차이이며, "High-Level" 차이는 클래스의 수, Object의 종류, 3D 위치와 Pose와 같은 기하학적 변형과 관련됨

- 도메인 불변 표현을 학습하는 새로운 방법인 **Domain Separation Networks (DSN)**을 제안함 

- 기존 연구는 주로 Source Domain의 표현을 Target Domain으로 매핑하거나 두 도메인 간에 공유되는 표현을 찾는 데 중점을둠

- 하지만 Shared된 표현이 Shared된 분포와 관련된 노이즈에 의해 오염될 수 있음

- 따라서  각 도메인에 Private한 하위 공간을 도입하여 도메인 고유의 속성(예: 배경 및 Low-Level 이미지 통계)을 캡처함

- 또한 Shared된 하위 공간은 Autoencoders와 명시적인 Loss Function을 통해 도메인 간에 공유되는 표현을 캡처함

- Private 하위 공간과 Shared 하위 공간이 서로 직교하도록 설정함으로써, 모델이 각 도메인의 고유한 정보를 분리하고, 이를 통해 더 의미 있는 표현을 생성할 수 있음

- 이 방법은 Object Classification과 Pose Estimation 작업을 위한 여러 데이터 세트에서 기존의 도메인 적응 기법보다 우수한 성능을 발휘하며, Private 및 Shared 표현의 시각화를 통해 해석 가능성을 높였음

## 2. Related Work

- Convolutional Neural Network(CNN) 기반  Unsupervised Domain Adaptation 중점을 두겠음 
- DANN 소개
- MMD 소개 (Deep Adaptation Network)
- CORAL 소개



## 3. Method

- 목표는 Source Domain에서 레이블이 있는 데이터 세트와 Target Domain에서 레이블이 없는 데이터 세트를 주면,  Source Domain에서 학습한 분류기가 Target Domain에서도 일반화되도록 하는 것

- 기존 연구처럼 모델은 Source Domain과 Target Domain의 이미지 표현이 유사해지도록 훈련됨

- 하지만 이러한 표현에는 종종 Shared된 표현과 강하게 연관된 노이즈가 포함될 수 있습니다 [REF]

- 최근 연구에서 제안된 **Shared-Space Component Analysis**에서 영감을 받아, DSN은 도메인 표현의 Private 및 Shared된 구성 요소를 명시적으로 그리고 공동으로 모델링함

- 표현의 Private 구성 요소는 도메인에 특화되며, Shared된 구성 요소는 두 도메인 모두에서 공유

- 모델이 이러한 분리된 표현을 생성하도록 유도하기 위해, 우리는 이 두 부분이 서로 독립적이 되도록 **Loss Function**을 추가

- 마지막으로, Private 표현이 여전히 유용하도록 하고 일반화 가능성을 높이기 위해 **Reconstruction Loss** 추가

- 모델은 두 도메인에서 유사한 Shared 표현과 각 도메인에 고유한 Private 표현을 생성

- 이로 인해 Shared 표현에 대해 훈련된 분류기는 두 도메인 간에 더 잘 일반화됨

### 3.1 Learning

DSN 모델에서의 추론은 다음과 같이 주어짐

	$\hat{x} = D(E_c(x) + E_p(x))$
	  
	$\hat{y} = G(E_c(x))$

- $\hat{x}$는 입력 $x$의 재구성이며, $\hat{y}$는 작업과 관련된 예측임

- 학습 목표는 다음 Loss를 최소화

$L = L_{task} + \alpha L_{recon} + \beta L_{difference} + \gamma L_{similarity}$

-  $\alpha$, $\beta$, $\gamma$는 Loss 항목 간의 trade-off para
 
- 분류 손실 $L_{task}$는 모델이 출력 레이블을 예측하도록 훈련

- Target Domain에는 레이블이 없다고 가정하므로,  Source Domain에만 적용

- Source Domain의 각 샘플에 대해 실제 클래스의 Negative Log-Likelihood를 최소화

    $L_{task} = - \sum y_{si} \cdot \log (\hat{y}_{si})$

$y_{si}$는 Source Domain 입력 $x_{si}$에 대한 클래스 레이블의 One-Hot Encoding이고, $\hat{y}_{si}$는 모델의 Softmax 예측 $\hat{y}_{si} = G(E_c(x_{si}))$.

Source 및 Target Domain 모두 적용되는 Scale-Invariant Mean Squared Error (MSE)를 재구성 손실 $L_{recon}$으로 사용:

$L_{recon} = \sum L_{si_{mse}}(x_{si}, \hat{x}_{si}) + \sum L_{si_{mse}}(x_{ti}, \hat{x}_{ti})$




- $L_{si_mse}(x, \hat{x})$는 다음과 같이 정의
	$L_{si\_mse}(x, \hat{x}) = \frac{1}{k} \| x - \hat{x} \|^2_2 - \frac{1}{k^2} ([x - \hat{x}] \cdot \mathbf{1}_k)^2$
	
- $k$는 입력 $x$의 픽셀 수이고, $\mathbf{1}_k$는 길이가 $k$인 벡터이며, $| \cdot |^2_2$는 제곱 $L_2$-노름

- **MSE Loss**는 전통적으로 재구성 작업에 사용되지만, 이는 스케일링 항목에 대해 올바른 예측을 하는 경우에도 벌점을 부과함

- 반면, Scale-Invariant MSE는 픽셀 간 차이를 벌점으로 부과하여 모델이 절대적인 색상이나 강도보다는 객체의 전반적인 모양을 학습하도록 함

- 실험 결과 전통적인 MSE 보다 좋았음

### 3.2 Similarity Losses

- Shared와 Private 인코더가 입력의 서로 다른 측면을 인코딩하도록 장려하는 **Difference Loss**는 두 도메인 모두에 적용

- 각 도메인의 Shared 및 Private 표현 사이의 Soft Subspace Orthogonality Constraint를 통해 손실 정의

- $H^s_c$와 $H^t_c$는 각각 Source 및 Target Data의 샘플에서 가져온 Shared된 표현의 행렬

- 마찬가지로 $H^s_p$와 $H^t_p$는 각각 Source 및 Target Data에서 가져온 Private 표현의 행렬

- **Difference Loss**는 각 도메인의 Shared 표현과 Private 표현 간의 직교성을 장려

	$L_{difference} = \| H^s_c{}^\top H^s_p \|^2_F + \| H^t_c{}^\top H^t_p \|^2_F$

-  $| \cdot |^2_F$는 **Frobenius Norm**의 제곱

- **Similarity Loss**는 Shared 인코더에서 나온 Source 및 Target의 Hidden 표현인 $h^s_c$와 $h^t_c$가 도메인에 상관없이 가능한 한 유사하게 유지되도록 장려 

- 우리는 두 가지 **Similarity Loss**를 실험했음

## 4. Evaluation

- Synthetic Dataset에서 모델을 학습하고, 노이즈가 있는 Real-World Dataset에서 테스트하는 문제에 동기부여를 받음

- Object Classification 데이터 세트를 사용하여 평가함

- MNIST와 MNIST-M, 독일 교통 표지판 인식 벤치마크(GTSRB)  그리고 Street View House Numbers (SVHN)

- Object Instance Recognition과 3D Pose Estimation의 표준인 Cropped LINEMOD 데이터 세트도 평가에 사용

- 다음과 같은 Unsupervised Domain Adaptation 시나리오에서 테스트
	(a) MNIST에서 MNIST-M으로
	(b) SVHN에서 MNIST로
	(c) Synthetic Traffic Signs에서 실제 GTSRB로
	(d) 검은 배경에 렌더링된 Synthetic LINEMOD Object Instances에서 실제 세계의 동일한 Object Instances로

- Unsupervised Domain Adaptation을 위한 하이퍼파라미터 최적화의 보편적인 방법을 발견하지 못했음

- 비교하는 모든 방법에서 동일한 프로토콜을 사용했으므로, 비교 수치는 공정하고 의미가 있음

### 4.2 Implementation Details

- initial learning rate was multiplied by 0.9 every 20, 000 steps
- batches of 32 samples from each domain for a total of 64 and the input
- images were mean-centered and rescaled to [−1, 1].
- In order to avoid distractions for the main classification task during the early stages of the training procedure, we activate any additional domain
- adaptation loss after 10, 000 steps of training. 

### 4.3 Discussion

- Domain Separation Networks는 MMD Regularization과 DANN을 모두 능가

- 특히 **DANN 손실**을 사용한 경우가 **MMD 손실**을 사용한 경우보다 더 나은 성능을 보임 (왜?)

- Soft Orthogonality Constraints ($L_{difference}$)가 학습에 미치는 영향을 조사함

- **DSN with DANN** 모델에서 이걸 제거한 경우, 모든 시나리오에서 성능이 일관되게 저하됨

- 또한, Scale-Invariant MSE 대신 일반적인 MSE를 사용하는 경우에도 성능이 일관되게 저하되는 것을 확인

- 마지막으로, Private 및 Shared 표현을 개별적으로 인코딩한 결과, MNIST와 MNIST-M 시나리오에서는 모델이 배경과 전경을 명확하게 분리하고, Source Domain과 매우 유사한 Shared 공간을 생성했음
## 5. Conclusion

- 기존 Unsupervised Domain Adaptation 기법을 개선한 딥러닝 모델을 제시

- 우리의 모델은 도메인 간 적응을 수행하면서도, 각 도메인에 Private한 표현과 두 도메인 간에 공유되는 Shared 표현을 명시적으로 분리함

- 기존 도메인 적응 기술을 사용하여 Shared 표현을 유사하게 만들고, Soft Subspace Orthogonality Constraints를 사용하여 Private 표현과 Shared 표현을 서로 다르게 유지함으로써, 우리의 모델은 다양한 Synthetic-to-Real 도메인 적응 시나리오에서 기존의 Unsupervised Domain Adaptation 기법을 능가하는 성능을 보임

- 또한, 모델이 학습한 Private 및 Shared 표현을 시각화할 수 있어, 도메인 적응 과정에 대한 해석 가능성을 제공함


