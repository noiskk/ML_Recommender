# 2024-2 Machine Learning
<br>

## Airbnb accommodation recommendations

## Dataset: Seattle Airbnb Open Data
https://www.kaggle.com/datasets/airbnb/seattle?select=listings.csv


## Preprocessing
1. 결측값 처리
  - name, host_location 등 일부 결측값이 남아있지만, 주요 분석에 영향을 주지 않거나 데이터를 손실하지 않기 위해 대부분의 결측값을 제거하거나 대체

2. 데이터형식 변환
  - host_since 등 날짜 데이터 datetime 형식으로 변환
  - price 숫자 형식으로 변환

3. 범주형 인코딩
  - 범주형(host_response_time, host_is_superhost, host_has_profile_pic, instant_bookable) 변수는 수치형으로 인코딩

4. 텍스트 데이터 정리
  - name과 같은 열은 분석에 필요하지 않다면 삭제
  - 필요한 열은 특정 단어를 토큰화하거나 길이를 계산하는 등의 텍스트 처리 기법을 사용

5. 새로운 컬럼 생성
  - 편의시설에 해당하는 열들은 1과 0으로 이진화, 편의 시설 개수를 합산한 새로운 열을 생성
  - host_since에서 호스트 경험 기간(예: 현재 날짜와의 차이)을 계산하여 추가 특성으로 활용

6. 이상치 처리
  - minimum_nights, maximum_nights, review_scores_* 등의 열에서는 극단적인 값(이상수치)을 상한/하한을 설정해 분석에 적합한 범위로 조정



## Modeling
1. Item-based Filtering
레시피-사용자 평점 행렬 생성 후 레시피 간의 코사인 유사도 계산

2. Content-based Filtering
사용자 프로필 만들고 유사도 계산

## Evaluation
1. Precision
- 추천한 숙소 중 실제 사용자가 선호한 숙소의 비율

2. Recall
- 사용자가 선호한 숙소 중 추천된 숙소의 비율


   

## 개선할점
하이브리드 방식
