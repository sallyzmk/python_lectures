---
# title: "혼공머 Chapter 01"
date: '2022-06-30 01:00'
---

# 0. 자율학습
- 파이썬 시각화 참고 블로그 (https://jehyunlee.github.io/)

## 파이썬 주요 라이브러리
- Machine Learning
  + 정형 데이터
  + 사이킷런(https://scikit-learn.org/stable/)

- Deep Learning
  + 비정형 데이터
  + Tensorflow(구글) vs Pytorch(페이스북)
  + 교재_혼공머: Tensorflow
  + R&D 연구: Pytorch (NumPy와 문법이 유사)
  + 실제 상용서비스: Tensorflow (안드로이드, App)


# Chapter 01

## 생선 분류1 도미 (45p)
- [데이터 다운로드](https://www.kaggle.com/datasets/aungpyaeap/fish-market)
- 도미, 곤들매기, 농어 등등
- 이 생선들을 프로그램으로 분류한다.


- 1. 30cm 도미라고 알려줘라


```python
fish_length = 31
if fish_length >= 30:
  print("도미")
```

    도미
    

- 2. 도미의 데이터 (47p)


```python
# 도미의 길이
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]

# 도미의 무게
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
```

## 데이터 가공 (도미)
- 여기서는 생략

## 도미 데이터 시각화
- 여러 인사이트 확인을 위해 시각화 및 통계 수치 계산
- 탐색적 자료 분석 (EDA : Exploratory Data Analysis)


```python
# 이 코드는 참고만 한다.
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


    
![png](images/Chapter01/images/Chapter01/output_11_0.png)
    


### 아래 코드를 사용해야되는 이유
- 파이썬 시각화는 객체지향으로 한다. 
- 이유: 시각화를 더 아름답게 다듬기 위해
- 캐글 시각화 참고할 때, 아래와 같이 하는 분들이 많다.


```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(bream_length, bream_weight)
ax.set_xlabel('length')
ax.set_ylabel('weight')
plt.show()
```


    
![png](images/Chapter01/output_13_0.png)
    


## 생선 분류2 빙어
- 1. 빙어 데이터 준비하기


```python
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```

## 데이터 가공 (빙어)
- 여기서는 생략

## 빙어 데이터 시각화


```python
fig, ax = plt.subplots()
ax.scatter(smelt_length, smelt_weight)
ax.set_xlabel('length')
ax.set_ylabel('weight')
plt.show()
```


    
![png](images/Chapter01/output_18_0.png)
    


## 도미+빙어 데이터 시각화 (50p)


```python
fig, ax = plt.subplots()
ax.scatter(bream_length, bream_weight)
ax.scatter(smelt_length, smelt_weight)
ax.set_xlabel('length')
ax.set_ylabel('weight')
plt.show()
```


    
![png](images/Chapter01/output_20_0.png)
    


## 모델링을 위한 데이터 가공
- 리스트 합치기
- 2차원 리스트 만들기
- 라벨링

- 두개의 리스트 합치기


```python
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
```

- 2차원 리스트로 만든다.


```python
fish_data = [[l,w] for l, w in zip(length, weight)]
# fish_data
fish_data[0:5]
```




    [[25.4, 242.0], [26.3, 290.0], [26.5, 340.0], [29.0, 363.0], [29.0, 430.0]]



### 라벨링
- 라벨링(=지도, 지도학습), y값(종속변수)가 필요하다.
- 문자를 숫자로 변경
  + 모델링(=알고리즘)은 문자 인식 불가, 숫자만 가능
- 도미인지 빙어인지 라벨링을 한다. (52p)




```python
# 리스트 연산자
# 모델링, 알고리즘은 문자 인식 불가, 반드시 숫자로 변경.
fish_target = [1] * 35 +[0] *14
print(fish_target)
```

    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    

## 모델링(=알고리즘)


```python
# 이 코드 말고 아래칸 코드 사용하기, 동일한 내용
# import sklearn
# model = sklearn.neighbors.KNeighborsClassifiter()
```

- 0. 패키지 설치, 실행


```python
# 패키지 설치
from sklearn.neighbors import KNeighborsClassifier

# 클래스 인스턴스화(불러오기)
kn = KNeighborsClassifier()
```

- 1. 모형 학습 (fit)


```python
#       독립변수, 종속변수
kn.fit(fish_data, fish_target)
```




    KNeighborsClassifier()



- 2. 모형이 잘 만들어졌는지 모형의 예측 정확도를 파악하기 (score)


```python
kn.score(fish_data, fish_target)
# [결과 해석] 1.0 = 100%
```




    1.0



- 3. 실제 예측 해보기 (predict)
  + 새로운 물고기 도착
  + 길이 : 30, 몸무게 : 60


```python
# fish_data가 2차원 데이터이기 때문에 대괄호 2개
kn.predict([[30, 600],[51,75],[77,85]])
# [결과 해석] 결과값이 1 이면 도미, 0 이면 빙어

# 모델에 넣어야 할 데이터가 많다면 -> 반복문 사용
```




    array([1, 0, 0])



- 4. 간단한 프로그램 만들기


```python
# int() : 숫자나 문자열을 정수형 데이터로 변환
ac_length = int(input("물고기 길이를 입력하세요..."))
ac_weigth = int(input("물고기 무게를 입력하세요..."))

preds = int(kn.predict([[ac_length, ac_weigth]]))
if preds == 1:
  print("도미")
else:
  print("빙어")
```

    물고기 길이를 입력하세요...14254
    물고기 무게를 입력하세요...123
    도미
    

## 실험 단계 (새로운 모델 제안)
- Default : 정확도 100%
- 새로운 모델 : 정확도 70%
- 여러 모델을 실험해 보고 가장 정확도가 높은 모델을 사용한다.

### 새로운 모델
- 하이퍼 파라미터 세팅
  + n_neighbors = 49 : 정확도 70%
- default(아무것도 세팅 안함) : 정확도 100%
- 결론 : 잘 모르면 어설프게 셋팅하지 말아라 -> 정확도가 떨어질 수 있음.



```python
kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)
```




    0.7142857142857143


