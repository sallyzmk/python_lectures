---
# title: "혼공머 Chapter 03"
date: '2022-06-30 03:00'
---

# 0. 자율학습
- 파이썬 시각화 참고 블로그 (https://jehyunlee.github.io/)
- 가장 최신의 알고리즘 논문 (https://paperswithcode.com/)

# Chapter 03-1 K-최근접 이웃 회귀
- 지도 학습 알고리즘은 크게 분류와 회귀
- 지도 학습 : 종속변수 존재
  + 분류 : 도미와 빙어 분류 문제 해결
  + 회귀 : 통계 회귀분석 y = ax + b , 수치예측


```python
# 패키지 설치
import numpy as np
print(np.__version__)
```

    1.21.6
    

- 전체 데이터


```python
perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )
```

- 시각화


```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(perch_length, perch_weight)
ax.set_xlabel('length')
ax.set_ylabel('weight')
plt.show()
```


    
![png](images/Chapter03/output_7_0.png)
    


- 모델링을 하기 위한 데이터 전처리 (여기서는 생략됨)
  + 2차원 리스트
  + 라벨링


```python
# 훈련 세트, 테스트세트를 (무작위 + 층화샘플링) 하여 분리함.
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
      perch_length, perch_weight, random_state = 42
)

train_input.shape, test_input.shape, train_target.shape, test_target.shape
```




    ((42,), (14,), (42,), (14,))



- 1차원 배열 -> 2차원 배열
  + NumPy reshape 사용


```python
# -1을 지정하면 나머지 원소 개수로 모두 채우라는 의미
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)
print(train_input.shape, test_input.shape)
print(train_input.ndim)
```

    (42, 1) (14, 1)
    2
    

## 결정계수
- Adjusted-R Squared : 회귀
- 정확한 지표
  + 대략적으로 근접하게 맞춘다.
- 0~1 사이의 지표, 1에 가까울수록 예측 모형의 완성도가 높은 것.


```python
 from sklearn.neighbors import KNeighborsRegressor

 knr = KNeighborsRegressor()
  
 # 모형 학습
 knr.fit(train_input, train_target)

 # 테스트 세트의 점수를 확인한다.
 # 테스트 세트에 있는 샘플을 정확하게 분류한 개수의 비율 =정확도
 print(knr.score(test_input, test_target))
```

    0.992809406101064
    


```python
test_input[0:5]
```




    array([[ 8.4],
           [18. ],
           [27.5],
           [21.3],
           [22.5]])




```python
from sklearn.metrics import mean_absolute_error

# 예측 데이터
test_prediction = knr.predict(test_input)
test_prediction[:5]
```




    array([ 60. ,  79.6, 248. , 122. , 136. ])




```python
# 실제 데이터
test_target[:5]
```




    array([  5.9, 100. , 250. , 130. , 130. ])




```python
# 테스트 세트에 대한 평균 절댓값 오차률
mae = mean_absolute_error(test_target, test_prediction)
print(mae)
# 실제 데이터 - 예측 데이터 -->> 음수를 정수로 바꿈 -->> 전체 오차를 더하고 평균을 냄
```

    19.157142857142862
    

- 예측이 평균적으로 19g 정도 다르다.
  + 확실한 건 오차가 존재한다.
  + 그러나 이 19g 정도가 의미 하는 것이 무엇이냐?
    + 19g은 절대 지표가 아님. 다른 결과와 비교해서 더 나은 것을 사용한다.
    + 더 많은 데이터를 수집 / 다른 모델을 사용
    + 개선 : 17g


## 과대적합 vs 과소 적합
- 매우 힘듬, 도망 가고 싶음
- 결과가 나온게 중요한 것이 아니라, 모형을 잘못 짠 것
- 과대적합: 훈련세트 점수 좋음, 테스트 점수 매우 안좋음
- 과소적합: 테스트 세트의 점수가 매우 좋음
- 결론: 뭔진 모르겠으나 제대로 모형이 훈련 안됨.
  + 모형을 서비스에 탑재 시킬 수 없음.


```python
print("훈련 평가", knr. score(train_input, train_target))
print("테스트 평가", knr.score(test_input,test_target))
```

    훈련 평가 0.9698823289099254
    테스트 평가 0.992809406101064
    

- 모형 개선


```python
# 이웃의 개수를 3으로 재 재지정
# 하이퍼 파라미터 : 여러 수를 대입해보고 가장 적합하게 나온 모형을 택한다. 3 이 가장 적합하게 나온 모형
knr.n_neighbors = 3

# 모형 다시 훈련
knr.fit(train_input,train_target)
print("훈련 평가", knr. score(train_input, train_target))
print("테스트 평가", knr.score(test_input,test_target))
```

    훈련 평가 0.9804899950518966
    테스트 평가 0.9746459963987609
    

## Chapter 03-1 복습


```python
import numpy as np
```


```python
perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )

print(perch_length.shape, perch_weight.shape)
```

    (56,) (56,)
    

### 데이터 가공


```python
# 훈련 스트와 테스트 세트로 나눈 후, 1차원 -> 2차원 배열로 변환
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    # 독립변수, 종속변수, 랜덤 고정 (데이터가 한개기 때문에 비율을 맞출 필요는 없다)
    perch_length, perch_weight, random_state= 42
)

train_input.shape, test_input.shape, train_target.shape, test_target.shape
```




    ((42,), (14,), (42,), (14,))




```python
# 1차원 -> 2차원 배열
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1,1)

# 배열을 바꿨을 때에는 항상 확인해주는 습관을 들이자
train_input.shape, test_input.shape
```




    ((42, 1), (14, 1))



### 데이터 시각화 -> 데이터 재가공
- 여기서는 생략

### 모델링


```python
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=3)

# 모형 훈련
knr.fit(train_input, train_target)
```




    KNeighborsRegressor(n_neighbors=3)



### 모델 평가
- 여기서는 생략

### 모델 예측
- 실제 서비스를 함


```python
# 농어의 길이 -->> 농어의 무게를 예측
print(knr.predict([[50]]))
# 결과가 뭔가 이상함. 원래 44에 1000이니 50은 더 커야되는데 1033.333밖에 안됨
```

    [1033.33333333]
    

### 모형 평가를 위한 시각화


```python
import matplotlib.pyplot as plt

# 50cm 농어의 이웃을 3개 (초록점)
distances, indexes = knr.kneighbors([[50]])

print(distances, indexes)

# 훈련 세트의 산점도를 그려보자
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D')

# 50cm 농어 데이터
plt.scatter(50,1033,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 비선형 KNN(K-최근접 이웃 회귀)의 약점
```

    [[6. 7. 7.]] [[34  8 14]]
    


    
![png](images/Chapter03/output_36_1.png)
    


### 결론 : 
  + 머신러능 = 알고리즘
  + 머신러닝/딥러닝 = 실험
  + 실험을 한다 = 다향한 방법을 써본다. = 다양한 알고리즘을 써본다.
  + [가장 최신의 알고리즘 논문](https://paperswithcode.com/)
  + 하루에도 수십개의 논문이 쏟아져 나오는데 어떻게 해서 몇개의 알고리즘이 선택을 받았을까?
  + 대중적인 몇개의 알고리즘만 기억
  + 전문가가 논문을 발표한 이후 캐글대회에서 실험해본다.
  + 캐글 대회 나가서 가장 최근 유행중인 알고리즘을 공부하는 것도 괜찮다. 현직자들은 바빠서 그렇게 못한다.

# Chapter 03-2 선형 회귀
- 가장 근접한 기울기를 함수를 통해 찾아내고 적용한다.


```python
# Python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

# 선형 회귀 모델을 훈련한다.
lr.fit(train_input, train_target)

# 50cm 농어에 대해 예측한다.
print(lr.predict([[50]]))
```

    [1241.83860323]
    


```python
print(lr.predict([[1000]]))
```

    [38308.12631868]
    

## 선형회귀의 모형
- 기울기, 절편이 궁금하다.



```python
print(lr.coef_, lr.intercept_)
# [결과 해석] 그래프가 음수에서부터 시작한다? 1cm짜리 농어의 무게가 -700얼마???
```

    [39.01714496] -709.0186449535477
    

## 다항회귀
- 선형회귀에서는 농어 1cm = -650g
- 직선의 기울기 대신, 곡선의 기울기를 쓰자
- 직선 = 1차 방정식, 곡선 = 2차 방정식
- $y = x^2 + ax + b$
- $y = a길이^2 + b길이 + 절편$


```python
# 140p
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

# 확인 출력
print(train_poly.shape, test_poly.shape)
print("---train_poly---")
print(train_poly[:3])
print("---test_poly---")
print(test_poly[:3])
```

    (42, 2) (14, 2)
    ---train_poly---
    [[384.16  19.6 ]
     [484.    22.  ]
     [349.69  18.7 ]]
    ---test_poly---
    [[ 70.56   8.4 ]
     [324.    18.  ]
     [756.25  27.5 ]]
    


```python
lr = LinearRegression()
lr.fit(train_poly,train_target)
print(lr.predict([[50 ** 2, 50]]))
```

    [1573.98423528]
    


```python
print(lr.coef_, lr.intercept_)
```

    [  1.01433211 -21.55792498] 116.0502107827827
    

- 무게 = 1.01 x $길이^2$ - 21.6 x 길이 + 116.05

다항회귀 모형 시각화


```python
# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만든다.
point = np.arange(15,50)

# 훈련 세트의 산점도를 그린다.
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D')

# 15에서 49까지 2차 방정식 그래프를 그린다.
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

# 50cm 농어 데이터
plt.scatter(50,1574,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


    
![png](images/Chapter03/output_49_0.png)
    



```python
# 모델 평가
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
```

    0.9706807451768623
    0.9775935108325122
    

# Chapter 03-3  [SKIP]
- 특성 공학과 규제
