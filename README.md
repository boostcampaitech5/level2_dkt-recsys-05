# lv2-RecSys-5 '스나이퍼'

## DKT (Deep Knowledge Tracing)

**프로젝트 개요**  

```
DKT는 Deep Knowledge Tracing의 약자로, 우리의 '지식 상태'를 추적하는 딥러닝 방법론입니다. 
DKT를 이용해 주어진 최종 문제를 학생이 맞출지 틀릴치 예측하는 이진 분류를 수행합니다.
```
<br>

**프로젝트 수행방법**

1. EDA & Feature Engineering
2. 모델 탐구
3. 모델 평가 및 개선
4. 결과 정리 및 분석

<br>

**데이터**

- train_data.csv
  - 행 : 2,266,586
  - 열 : 6개 - `userID`, `assessmentItemID`, `testID`, `answerCode`, `Timestamp`, `KnowledgeTag`
- test_data.csv
  - 행 : 744 (train user 수의 10%)
  - 열 : train_data와 동일. 단, `answerCode` 값은 모두 `-1`

<br>

## EDA & Feature Engineering
Feature Engineering 결과는 LGBM 모델에서 주로 사용하였다.

### 1. EDA

- train_data과 test_data 각각 EDA를 수행하고 비교하며 탐색하였다.
- 파악한 주요 특징은 다음 표와 같다. (자세한 EDA과정을 보고 싶다면 `EDA.ipynb` 파일 참고)

| column | EDA |
| :-------: | :-----------: |
| `userID` | 대부분 250문항 이하로 문제를 풀었고, train과 test의 정답률은 비슷함 |
| `assessmentItemID` | 후반으로 갈수록 정답률이 떨어짐 |
| `testID` | test_first(두 번째 번호)와 test_rest(마지막 3자리)가 각각 시험지의 난이도(정답률)와 생성일자와 연관이 있음 |
| `Timestamp` | 시기별, 요일별, 시간별 정답률에서 다른 양상을 보임 |
| `KnowledgeTag` | 태그별 빈도수 상이함. 태그의 빈도는 정답률과 큰 상관이 없어보임 |

<br>

### 2. Feature Engineering

- EDA를 바탕으로 약 23여개의 feature들을 생성하였다.
- 기본적으로 리더보드에서 성능 향상이 있는지를 기준으로 최적의 Feature 조합을 찾아갔다.
- `SHAP`와 `Feature 중요도`값도 참고하였다.

- 생성 feature 목록
  - 사용자별 누적 집계 Feature
    - user_test_correct_answer : user별 각 테스트 내에서 맞힌 문항의 개수
    - user_test_total_answer : user별 각 테스트 내에서 푼 문항의 개수
    - user_test_acc : user별 각 테스트 내 정답률
    - user_tag_correct_answer : user별 각 태그(태그에 해당하는 문항)를 맞힌 개수
    - user_tag_total_answer : user별 각 태그를 푼 개수
    - user_tag_acc : user별 각 태그 내 정답률
    - user_tag_incorrect : 유저별 각 태그를 틀린 개수
    - user_tag_inacc : 유저별 각 태그 오답률

  - ID
    - testid_first : testid의 3번째 자리 수
    - testid_rest : testid의 마지막 3자리수
    - itemseq : itemid의 마지막 자리 수(문항 순서)

  - 평균
    - item_mean : 문항별 평균 정답률
    - test_mean : 테스트별 평균 정답률
    - tag_mean : 태그별 평균 정답률
    - item_std : 문항별 정답률의 표준편차
    - test_std : 테스트별 정답률의 표준편차
    - tag_std : 태그별 정답률의 표준편차
  
  - TimeStamp
    - month : 문항 풀이한 달
    - hour : 문항 풀이한 시간
    - repeat : 테스트별 반복 풀이한 횟수
    - elapse : 이전 문항을 푼 시간
    - total_elapse : 테스트 내 누적 문항 풀이 시간
    - encoded_time : 유저별 처음 문제를 풀이한 일부터 지난 일수

<br>


## 모델
크게 3가지 계열(ML 계열, Transformer 계열, GNN 계열)의 모델을 사용해 결과를 냈다.

### 1. 모델 탐구
(주요 탐구 대상 **굵게**처리)
- ML 계열 : **LGBM(LightGBM)**, Catboost, XGBoost
- Transformer 계열 : **BERT**, **Saint+**, XLnet, ELECTRA, LastQuery
- GNN 계열 : **LGCN(LightGCN)**

<br>

### 2. 모델 개선

- LGBM
  - Feature Engineering의 영향을 가장 많이 받음
- BERT
  - 새로운 토큰을 임베딩하는 방식으로 Feature Engineering을 활용하려 하였으나 결과 개선에 효과를 보지 못해 기존 Feature 그대로 사용
- Saint+
  - 학생이 '문제를 푸는 데 소요되는 시간'과 '문제 사이 간격'이란 Feature를 추가
  - 적은 수의 Feature를 넣고 Transformer에 학습을 맡기는 것이 더 효과적임
- LGCN
  - Epoch 수 : 증가
  - Layer 수 : 1개 - layer 수를 늘렸을 때, valid auc는 증가하는 것과는 달리 리더보드상 auc는 저하되는 현상이 발행함

<br>

### 3. 모델 선정 및 결과

- 선정 모델
  ```
  LGBM, BERT, Saint+, LGCN, XLnet, ELECTRA, LastQuery
  ```
- 결과 앙상블

<br>

### 4. 결론

- 단일 모델로 첫번째는 `Saint+`이 두번째는 `LGBM`이 좋은 결과를 보였다.
- 전체 데이터셋(Train과 Test 합친 데이터셋)으로 최적화 작업을 진행했다면 리더보드 상 결과가 더 개선될 것으로 예상된다.
- 과적합의 원인이라 판단했던 Feature들의 재평가가 필요하다.


<br>

## 최종 제출
```
LGBM 0.25 + Transformer 0.7(Saint+ 0.15) + LGCN 0.05  
```
- 숫자는 앙상블 시 비율을 나타낸다

<br>
