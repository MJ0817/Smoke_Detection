

---
# AI 연기 감지 모델을 활용한 스마트 화재 예측 시스템
<img width="768" alt="스크린샷 2024-08-21 오후 10 51 38" src="https://github.com/user-attachments/assets/4c18fa78-79a6-4a1b-bec0-34cfd2fdfee5">

## 프로젝트 개요
이 프로젝트의 목적은 IoT(사물인터넷) 장치로부터 수집된 데이터를 활용하여 AI 기반 연기 감지 모델을 개발하고, 이를 통해 보다 스마트하고 빠르게 화재를 예측할 수 있는 시스템을 구축하는 것입니다. 이를 통해 화재를 조기에 감지하고 신속하게 대응함으로써 인명과 재산 피해를 최소화하고, 공공 안전을 증진하고자 합니다.

## 주요 기능
- **실시간 연기 감지**: IoT 센서 데이터를 실시간으로 분석하여 화재의 징후를 빠르게 감지합니다.
- **AI 모델을 활용한 예측**: 다양한 AI 알고리즘을 활용하여 높은 정확도로 화재 발생 가능성을 예측합니다.
- **다양한 환경에서의 적용**: 실내외 다양한 환경에서의 연기 및 화재 예측이 가능합니다.

## 데이터 수집 및 전처리
![스크린샷 2024-08-27 153428](https://github.com/user-attachments/assets/62a8abca-3e90-440e-8e0b-87f930e88378)

- **데이터 수집**: IoT 장치를 통해 62,630개의 데이터셋을 수집하였으며, 여기에는 온도, 습도, TVOC(총휘발성유기화합물), CO2 농도, 센서가 감지한 수소와 에탄올 농도, 공기압, 미세먼지 농도 등의 다양한 환경 변수가 포함됩니다.
- 
- **데이터 전처리**: 데이터의 품질을 높이기 위해 결측값 처리, 중복 데이터 제거, 데이터 정규화 등의 전처리 과정을 거쳤습니다.

## 사용된 기술
- **프로그래밍 언어**: Python
- **라이브러리 및 프레임워크**: NumPy, pandas, matplotlib, seaborn, scikit-learn, TensorFlow, XGBoost
- **모델링**: 
  - **로지스틱 회귀(Logistic Regression)**: 이진 분류 문제에 효과적이며, 해석이 쉬운 모델입니다.
  - **가우시안 나이브 베이즈(Gaussian Naive Bayes)**: 연속형 데이터가 정규 분포를 따른다고 가정하여 사용합니다.
  - **베르누이 나이브 베이즈(Bernoulli Naive Bayes)**: 이진 데이터를 처리하는 데 적합한 모델입니다.
  - **서포트 벡터 머신(Support Vector Machine)**: 분류 성능이 뛰어난 모델로, 특히 고차원 공간에서 유용합니다.
  - **랜덤 포레스트(Random Forest)**: 여러 결정 트리의 앙상블로, 높은 정확도와 과적합 방지에 효과적입니다.
  - **K 최근접 이웃(K-Nearest Neighbors)**: 간단하지만 데이터의 분포를 잘 반영하는 비모수적 방법입니다.
  - **극단적 그라디언트 부스팅(Extreme Gradient Boosting)**: 높은 성능을 자랑하는 부스팅 알고리즘입니다.
  - **인공신경망(Neural Network)**: 복잡한 패턴 인식에 강력한 모델로, 이 프로젝트에서 가장 높은 성능을 기록했습니다.

## 모델 성능 평가
- **성능 지표**: 모델의 성능은 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-Score 등의 다양한 지표를 사용하여 평가되었습니다.
- **주요 결과**: 인공신경망(Neural Network) 모델이 가장 높은 성능을 보였으며, 정확도 95% 이상을 기록했습니다.
- **모델 성능 비교**: 각 모델의 혼동 행렬(Confusion Matrix)과 성능 비교 그래프를 통해 모델의 예측 능력을 시각적으로 확인할 수 있습니다.
![Figure_1](https://github.com/user-attachments/assets/6fde022c-dc2a-40ea-9048-2a1ccc65de01)


## 사용법
1. **데이터 준비**: IoT 장치로부터 수집된 데이터를 준비합니다.
2. **모델 훈련**: 제공된 Python 스크립트를 사용하여 데이터를 전처리하고 AI 모델을 훈련시킵니다.
3. **모델 평가**: 모델의 성능을 평가하고, 필요에 따라 파라미터를 조정합니다.
4. **예측 실행**: 훈련된 모델을 사용하여 실시간으로 화재 발생 가능성을 예측합니다.

## 한계점
- **데이터 다양성의 부족**: 현재 모델은 특정 환경과 조건에서 수집된 데이터로 훈련되었습니다. 더 다양한 환경에서의 데이터가 필요합니다.
- **모델 복잡성 및 실행 시간**: 일부 고성능 모델, 특히 인공신경망(Neural Network)은 많은 계산 자원을 필요로 하며, 실시간 예측에는 속도 개선이 필요합니다.
- **센서 데이터의 신뢰성**: IoT 장치에서 수집된 센서 데이터의 정확도가 모델 성능에 직접적인 영향을 미칩니다. 센서의 정확도와 유지 보수가 중요합니다.
- **환경 변수의 변화**: 온도, 습도 등 환경 변수는 시간과 조건에 따라 변동성이 큽니다. 이를 효과적으로 반영하기 위한 데이터 업데이트가 필요합니다.


---

