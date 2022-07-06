---
title: "혼공머 Chapter 06"
date: '2022-07-05 01:00'
---

# 0. 자율학습
- 실무에서의 난이도
  + 비지도학습 >> 지도학습
- 비지도학습 : 분류, 수치적으로 분류, 특성, 분포 등으로 1차적으로 분류, 사람이 뭐가 뭔지 지정해줌.
- 지도학습 : 이게 사과야!, 이진분리
- 알고리즘은 쉬우나, 사람이 정의를 내려주기 애매 해서 실무에서의 난의도가 높다.
  + 러시아-우크라이나
  + 세계뉴스? 전쟁뉴스? 경제뉴스?

# Chapter 06-1 군집 알고리즘
- 

# Chapter 06-2 K-평균
- 

# Chapter 06-3 주성분 분석
- 이론적으로는 어려움
  + 좌표계 공간 개념
  + 공분산_통계관련 내용
  + 직교 + 회전
- Feature Engineering 기법
- StandatdScaler()
- 현 ML의 문제점 컬럼의 갯수가 매우 많음. (데이터 수집하는 방법이 다양해졌기 때문_ 음성, 사진, 위치 등 - google analytics)


## 차원과 차원 축소
- PCA : 차원(=데이터 특성=컬럼)을 축소할 수 있음.
- 특성이 많으면 훈련 데이터에 쉽게 과대적합된다.
- 특성을 줄여서 학습 모델의 성능을 향상 및 학습 시간을 감소시킨다.
- 대표적인 방법론 : PCA, EFA

### PCV vs EFA
- EFA(탐색적 요인 분석),Factor Analysis
  + 예) 국어, 수학, 과학, 영어
  + 예) 국어 40, 수학 100, 과학 100, 영어 30 / 귀 학생의 언어영역은 수준이 낮은 편이나 수리영역은 매우 수준이 높습니다.
  + 비슷한 성질끼리 묶어주는 것
  + 주로 범주형 & 수치 데이터셋

- PCA (주성분 분석)
  + 장비1, 장비2, 장비3, 장비4, ...
  + PC1, PC2, PC3, PC4, ..PCN   <- PCV값
  + PCV값만 보고 원래 가지고 있던 정보를 알 수 없음. (정보손실)
  + 범주형 데이터셋에는 사용 X
  + 무조건 수치형 데이터에서만 사용
  + PCA 실행 전 반드시 표준화 처리(스케일링 실행, 단위 맞추기)


## 주성분 분석 (320p)
- 여러개 데이터 특성을 소수의 데이터 특성으로 줄인다-> 그럼에도 불구하고 원본 데이터의 특성을 잘 나타낸다.

## PCA 클래스


```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy  # -O 숫자 0이 아닌 영문 O
```

    --2022-07-05 04:58:55--  https://bit.ly/fruits_300_data
    Resolving bit.ly (bit.ly)... 67.199.248.11, 67.199.248.10
    Connecting to bit.ly (bit.ly)|67.199.248.11|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://github.com/rickiepark/hg-mldl/raw/master/fruits_300.npy [following]
    --2022-07-05 04:58:55--  https://github.com/rickiepark/hg-mldl/raw/master/fruits_300.npy
    Resolving github.com (github.com)... 140.82.113.3
    Connecting to github.com (github.com)|140.82.113.3|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fruits_300.npy [following]
    --2022-07-05 04:58:55--  https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fruits_300.npy
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3000128 (2.9M) [application/octet-stream]
    Saving to: ‘fruits_300.npy’
    
    fruits_300.npy      100%[===================>]   2.86M  --.-KB/s    in 0.07s   
    
    2022-07-05 04:58:56 (42.6 MB/s) - ‘fruits_300.npy’ saved [3000128/3000128]
    
    


```python
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)

# 300개 행, 10000개의 열
fruits_2d.shape
```




    (300, 10000)



- pca


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(fruits_2d)
```




    PCA(n_components=50)




```python
print(pca.components_.shape)
# n_components=50으로 지정했기 때문에 pca.components_ 배열의 첫 번째 차원이 50입니다. 즉 50개의 주성분을 찾음.
# 두 번째 차원은 항상 원본 데이터의 특성 개수와 같은 10,000입니다.
```

    (50, 10000)
    


```python
import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr)    # n은 샘플 개수입니다
    # 한 줄에 10개씩 이미지를 그립니다. 샘플 개수를 10으로 나누어 전체 행 개수를 계산합니다. 
    rows = int(np.ceil(n/10))
    # 행이 1개 이면 열 개수는 샘플 개수입니다. 그렇지 않으면 10개입니다.
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, 
                            figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:    # n 개까지만 그립니다.
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()
```


```python
draw_fruits(pca.components_.reshape(-1, 100, 100))
```


    
![png](output_14_0.png)
    





```python
# 중요!!!! 머신러닝에서 컬럼의 갯수를 10,000개에서 50로 줄임. 다만 수치데이터만 가능!!
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
```

    (300, 50)
    

- 훈련데이터, 테스트 데이터 분리

## 설명된 분산
- 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값


```python
# 주성분이 원본 데이터의 분산을 92% 정도 나타냈다.
# 원본 이미지 압축

print(np.sum(pca.explained_variance_ratio_))
```

    0.9215454602209046
    


```python
plt.plot(pca.explained_variance_ratio_)
plt.show()
# [결과 해석] 50개도 여전히 많다.
```


    
![png](output_20_0.png)
    



```python
print(np.sum(pca.explained_variance_ratio_[:30]))
```

    0.8783777973933925
    


```python
print(np.sum(pca.explained_variance_ratio_[:51]))
```

    0.9215454602209046
    

## 다른 알고리즘과 함꼐 사용하기  [SKIP]
- 과일 사진 원본 데이터와 PCA로 축소한 데이터를 지도 학습에 적용해 보고 어떤 차이가 있는지 알아보자.
- 사이킷런의 LogisticRegression 모델과 비교


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
```

- 지도 학습 모델을 사용하려면 타깃값이 있어야 한다.
- 사과 0, 파인애플 1, 바나나 2


```python
target = np.array([0]*100+[1]*100 + [2]*100)
```

- 먼저 원본 데이터인 fruits_2d를 사용
- 로지스틱 회귀 모델에서의 성능을 가늠해 보기 위해 cross_validte()로 교차 검증 수행


```python
from sklearn.model_selection import cross_validate
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
```

    0.9966666666666667
    1.5436992645263672
    

- PCA로 축소한 fruits_pca 사용


```python
scores = cross_validate(lr,fruits_pca,target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
```

    1.0
    0.0634692668914795
    

- [결과 해석] 50개의 특성만 사용 했음에도 정확도가 100%, 훈련시간도 적어짐

- 주성분 대신 원하는 설명된 분산의 비율을 입력해도 된다.
- 0~1 사이의 비율을 실수로 입력 


```python
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
```




    PCA(n_components=0.5)




```python
print(pca.n_components_)
```

    2
    

- [결과 해석] 단 2개의 특성만으로 원본 데이터에 있는 분산의 50%를 표현할 수 있다.

- 이 모델을 원본 데이터로 변환
- 주성분이 2개 이므로 변환된 데이터 크기는 (300,2)


```python
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
```

    (300, 2)
    

- 2개의 특성만 사용해도 교차검증의 결과가 좋을까?


```python
scores = cross_validate(lr,fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))
```

    0.9933333333333334
    0.04194316864013672
    

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
    

- [결과해석] 99%의 정확성

- 차원 축소된 데이터를 이용해 K-평균 알고리즘으로 클러스터를 찾아보자.


```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))
```

    (array([0, 1, 2], dtype=int32), array([110,  99,  91]))
    

- 이미지 출력


```python
for label in range(0,3):
  draw_fruits(fruits[km.labels_==label])
  print("/n")
```


    
![png](output_44_0.png)
    


    /n
    


    
![png](output_44_2.png)
    


    /n
    


    
![png](output_44_4.png)
    


    /n
    

- [결과 해석] 2절에서 찾은 클러스터와 비슷하게 파인애플은 사과와 조금 혼돈되는 면이 있다.

- 시각화
  + 3개 이하로 차원을 줄이면 화면에 출력하기 비교적 쉽다.
  + fruits_pca 데이터는 2개의 특성이 있기 때문에 2차원으로 표현 할 수 있다.


```python
for label in range(0,3):
  data = fruits_pca[km.labels_==label]
  plt.scatter(data[:,0], data[:,1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()
```


    
![png](output_47_0.png)
    

