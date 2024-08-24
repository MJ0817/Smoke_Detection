

---

# AI 연기 감지 모델을 활용한 스마트 화재 예측 시스템

![DALL·E 2024-08-21 22 50 05 - A high-tech smoke detection system integrated with AI, depicted in a futuristic environment  The image shows a modern smoke detector with digital inte](https://github.com/user-attachments/assets/3bcdc03b-2c3c-4764-8673-d39638423257)



## 프로젝트 개요

이 프로젝트의 목적은 IOT(사물인터넷) 장치로부터 수집된 데이터를 활용하여 AI 기반 연기 감지 모델을 개발하고, 이를 통해 보다 스마트하고 빠르게 화재를 예측할 수 있는 시스템을 구축하는 것입니다. 이를 통해 화재를 조기에 감지하고 신속하게 대응함으로써 인명과 재산 피해를 최소화하고, 공공 안전을 증진하고자 합니다.

## 주요 기능

- **실시간 연기 감지**: IOT 센서 데이터를 실시간으로 분석하여 화재의 징후를 빠르게 감지합니다.
- **AI 모델을 활용한 예측**: 다양한 AI 알고리즘을 활용하여 높은 정확도로 화재 발생 가능성을 예측합니다.
- **다양한 환경에서의 적용**: 실내외 다양한 환경에서의 연기 및 화재 예측이 가능합니다.

## 데이터 수집 및 전처리

- **데이터 수집**: IOT 장치를 통해 62,630개의 데이터셋을 수집하였으며, 여기에는 온도, 습도, TVOC(총휘발성유기화합물), CO2 농도, 센서가 감지한 수소와 에탄올 농도, 공기압, 미세먼지 농도 등의 다양한 환경 변수가 포함됩니다.
- **데이터 전처리**: 데이터의 품질을 높이기 위해 결측값 처리, 중복 데이터 제거, 데이터 정규화 등의 전처리 과정을 거쳤습니다.

## 사용된 기술

- **프로그래밍 언어**: Python
- **라이브러리 및 프레임워크**: NumPy, pandas, matplotlib, seaborn, scikit-learn, TensorFlow, XGBoost
- **모델링**: 로지스틱 회귀(Logistic Regression), 가우시안 나이브 베이즈(Gaussian Naive Bayes), 버누리 나이브 베이즈(Bernoulli Naive Bayes), 서포트 벡터 머신(Support Vector Machine), 랜덤 포레스트(Random Forest), K 최근접 이웃(K-Nearest Neighbors), 극단적 그라디언트 부스팅(Extreme Gradient Boosting), 인공신경망(Neural Network)

## 모델 성능 평가

- **성능 지표**: 모델의 성능은 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-Score 등의 다양한 지표를 사용하여 평가되었습니다.
- **주요 결과**: 인공신경망(Neural Network) 모델이 가장 높은 성능을 보였으며, 정확도 95% 이상을 기록했습니다.
- **모델 성능 비교**: 각 모델의 혼동 행렬(Confusion Matrix)과 성능 비교 그래프를 통해 모델의 예측 능력을 시각적으로 확인할 수 있습니다.

## 사용법

1. **데이터 준비**: IOT 장치로부터 수집된 데이터를 준비합니다.
2. **모델 훈련**: 제공된 Python 스크립트를 사용하여 데이터를 전처리하고 AI 모델을 훈련시킵니다.
3. **모델 평가**: 모델의 성능을 평가하고, 필요에 따라 파라미터를 조정합니다.
4. **예측 실행**: 훈련된 모델을 사용하여 실시간으로 화재 발생 가능성을 예측합니다.



## 향후 연구 방향

- **데이터셋 확장**: 다양한 환경과 조건에서의 데이터 수집을 확대하여 모델의 일반화를 향상시킬 예정입니다.
- **모델 경량화 연구**: 실시간 예측을 위한 경량화 모델을 개발하여, 더욱 빠른 화재 감지를 가능하게 할 예정입니다.
- **AI 모델 통합**: 다른 AI 기술과의 통합을 통해 연기 감지 시스템을 더욱 고도화하고, 공공 안전을 증진하는 데 기여할 계획입니다.


---

이 README 파일은 프로젝트의 목적과 기능, 사용 방법 등을 명확하게 설명하며, 사용자가 쉽게 이해하고 프로젝트에 참여할 수 있도록 작성되었습니다.