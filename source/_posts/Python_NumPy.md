---
# title: "Python NymPy"
date: '2022-06-28 01:00'
---

## NumPy 라이브러리 설치방법
[NumPy 라이브러리 설치방법](https://www.notion.so/NumPy-b76659fce0554312bb6257d5a1b97d5d)

## NumPy란?
- NumPy: 수치계산, 행렬 연산. 예측.
- .array : 리스트를 행렬로 바꿔줌.
- 이미지도 수치로 표현할 수 있다.
- .reshape : 행렬의 배열을 바꿔줌 (2:3 -> 3:2)

- 내장 모듈(=라이브러리= 패키지) (X)
- 별도 라이브러리 설치 필요
- 구글코랩에서는 별도 라이브러리 설치 불필요


```python
import numpy as np
print(np.__version__)
```

    1.21.6
    


```python
temp = [1, 2, 3]  # 리스트
temp_array = np.array(temp) # 리스트에서 배열로 변환

print(type(temp))
print(type(temp_array))
```

    <class 'list'>
    <class 'numpy.ndarray'>
    


```python
# 사칙연산
math_score = [90, 80, 100]
eng_score = [80, 90, 100]

math_score + eng_score  # 리스트가 합쳐짐
math_array = np.array(math_score)
eng_array = np.array(eng_score)

total = math_array + eng_array
print(total)
print(type(total))

print(np.min(total)) # 최소
print(np.max(total))  # 최대
print(np.sum(total))  # 평균

```

    [170 170 200]
    <class 'numpy.ndarray'>
    170
    200
    540
    

## 차원 확인
- 배열의 차원 확인 필요


```python
# 1차원 배열 생성
temp_arr = np.array([1,2,3])
print(temp_arr.shape) # 몇 차원인지 확인 (3,) <- 1차원 형태인 값이 3개 있다.
print(temp_arr.ndim)  # 몇 차원인지 확인
print(temp_arr)
```

    (3,)
    1
    [1 2 3]
    


```python
# 2차원 배열 생성
temp_arr = np.array([[1,2,3],[4,5,6]])  # 짝을 맞춰줘야함. 3개3개 4개4개. 수가 없으면 NA라도 표시.
print(temp_arr.shape) # (2,3) <- 2차원 형태 2개, 2차원 형태 안에 값이 3개
print(temp_arr)
```

    (2, 3)
    [[1 2 3]
     [4 5 6]]
    


```python
# 3차원 배열 (이미지)
temp_arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(temp_arr.shape) # (2,2,3) <- 수를 다 곱하면 전체 값이 대략 몇개 들었는지 알 수 있다.
print(temp_arr.ndim)
print(temp_arr)
```

    (2, 2, 3)
    3
    [[[ 1  2  3]
      [ 4  5  6]]
    
     [[ 7  8  9]
      [10 11 12]]]
    

## 배열 생성의 다양한 방법들
- 모두 0으로 채운다


```python
import numpy as np
print(np.__version__)
```

    1.21.6
    


```python
temp_arr = np.zeros((3,3,3))  # 사용자가 차원 지정
temp_arr
```




    array([[[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]],
    
           [[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]]])




```python
# 모두 1로 채운다
temp_arr = np.ones((2,3))
temp_arr
```




    array([[1., 1., 1.],
           [1., 1., 1.]])




```python
# 임의의 상수값으로 채운다
temp_arr = np.full((3,3),5)
temp_arr
```




    array([[5, 5, 5],
           [5, 5, 5],
           [5, 5, 5]])



- 최소, 최대 숫자의 범위를 정해두고, 각 구간별로 값을 생성


```python
temp_arr = np.linspace(5, 10, 10)  # 5~10 사이의 값을 10개 가지고 와라
temp_arr
```




    array([ 5.        ,  5.55555556,  6.11111111,  6.66666667,  7.22222222,
            7.77777778,  8.33333333,  8.88888889,  9.44444444, 10.        ])



- 반복문 사용시, 자주 등장하는 배열


```python
temp_arr = np.arange(1, 11, 1)
temp_arr
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])



## 난수 생성


```python
# 방법 1
from numpy import random # numpy라는 클래스 안에 random이라는 
x=random.rand()
print(x)

# 방법 2
import numpy
x= numpy.random.rand()
print(x)
```

    0.8979748928243543
    

- 랜덤 정수값 추출


```python
from numpy import random
# x= random.randint(100, size = (5)) # 100 <- 상한가
x= random.randint(100, size = (3, 5)) # 배열도 설정가능
print(x)
```

    [[81 50 84 85 17]
     [ 9 32 85  8 23]
     [64 30  6 14 47]]
    

- 랜덤 배열, 실숫값 추출


```python
from numpy import random
x= random.rand(2,5)
print(x)
print(type(x))
```

    [[0.4493624  0.03890092 0.11260887 0.0690728  0.88457657]
     [0.52812858 0.3770578  0.9160059  0.32855566 0.09977088]]
    <class 'numpy.ndarray'>
    

## NumPy 사칙 연산 


```python
import numpy as np
array_01 = np.array([1,2,3])
array_02 = np.array([10,20,30])

# 덧셈
newArr = np.add(array_01, array_02)
print(newArr)

# 뺄셈
newArr = np.subtract(array_01, array_02)
print(newArr)

# 곱하기
newArr = np.multiply(array_01, array_02)
print(newArr)


# 나누기
newArr = np.divide(array_01, array_02)
print(newArr)

# 거듭제곱
array_01 = np.array([1,2,3])
array_02 = np.array([2,2,2])
newArr = np.power(array_01, array_02)
print(newArr)

```

    [11 22 33]
    [ -9 -18 -27]
    [10 40 90]
    [0.1 0.1 0.1]
    [1 4 9]
    

## 소숫점 정렬
- 소숫점을 정렬하는 다양한 방법 (np.trunc, np.fix)


```python
# 소수점 제거 (np.trunc, np.fix)
import numpy as np
temp_arr =np.trunc([-1.23,1.23])
print(temp_arr)

temp_arr =np.fix([-1.23,1.23])
print(temp_arr)
```

    [-1.  1.]
    [-1.  1.]
    


```python
# 임의의 소숫점 자리에서 반올림 (np.around)
temp_arr= np.around([-1.257438763, 1.232457654], 3)
print(temp_arr)
```

    [-1.257  1.232]
    


```python
# 소숫점 모두 내림 (np.floor)
temp_arr = np.floor([-1.225345343, 1.23546546])
print(temp_arr)
```

    [-2.  1.]
    


```python
# 소숫점 모두 올림 (np.ceil)
temp_arr = np.ceil([-1.3536331313,1.2768768])
print(temp_arr)
```

    [-1.  2.]
    

## 조건식
- pandas 가공
- numpy 와 섞어서 사용
- 조건식
  + 하나의 조건식 (np.where)
  + 다중 조건식 (np.select)


```python
temp_arr =np.arange(10)
temp_arr
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# 5보다 작으면 원 값 유지

# 5보다 크면 기존값 곱하기 10을 해줌.
```


```python
# np.where(조건식, 참일 때, 거짓일 때)
np.where(temp_arr <5, temp_arr, temp_arr *10)
```




    array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])




```python
temp_arr =np.arange(10)
# temp_arr
cond_list = [temp_arr>5, temp_arr <2]
choice_list = [temp_arr *2, temp_arr +100]

# np.select(조건식 리스트, 결괏값 리스트, default = )
np.select(cond_list,choice_list,default  = temp_arr)
```




    array([100, 101,   2,   3,   4,   5,  12,  14,  16,  18])



## Reshape
- 배열의 차원 또는 크기를 바꾼다.
- 곱셈만 할 줄 알면 끝.


```python
import numpy as np
temp_array = np.ones((3,4))
print(temp_array.shape)
print(temp_array)
```

    (3, 4)
    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]]
    


```python
# after_reshape = temp_array.reshape(1, 12)
# after_reshape = temp_array.reshape(2,6)
after_reshape = temp_array.reshape(2,2,3)
after_reshape
print(after_reshape.shape)
print(after_reshape)
```

    (2, 2, 3)
    [[[1. 1. 1.]
      [1. 1. 1.]]
    
     [[1. 1. 1.]
      [1. 1. 1.]]]
    


```python
after_reshape = temp_array.reshape(2,2,-1) # 사이즈가 너무 커서 곱셈 계산이 안되면, 자동 계산 하게끔 만들 수 있다. '-1'
after_reshape
print(after_reshape.shape)
print(after_reshape)
```

    (2, 2, 3)
    [[[1. 1. 1.]
      [1. 1. 1.]]
    
     [[1. 1. 1.]
      [1. 1. 1.]]]
    

## 브로드 캐스팅
- Python 데이터 분석 강의안_220423.pdf 43p

## pandas 튜토리얼


```python
import pandas as pd
print(pd.__version__)

```

    1.3.5
    


```python
temp_dict = {
    'col1' : [1, 2],
    'col2' : [3, 4]
}

df = pd.DataFrame(temp_dict)  # 가상의 데이터 프레임을 만듬.
print(df)
print(type(df))
```

       col1  col2
    0     1     3
    1     2     4
    <class 'pandas.core.frame.DataFrame'>
    

## 구글 드라이브 연동
- [판다스 10분 완성](https://dataitgirls2.github.io/10minutes2pandas/)
- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [파이썬 데이터 클리닝 쿡북_파이썬과 판다스를 활용한 데이터 전처리](http://www.yes24.com/Product/Goods/105030647)


```python
from google.colab import drive
drive.mount('/content/drive')

```

    Mounted at /content/drive
    


```python
DATA_PATH = '/content/drive/MyDrive/Colab Notebooks/human_AI/Basic/Chapter 3. pandas/data/'
lemonade = pd.read_csv(DATA_PATH + 'Lemonade2016.csv')
# covid_data = pd.read_csv(DATA_PATH + 'owid-covid-data.csv')

lemonade.info() # 데이터가 잘 드러왔는지 확인 str()과 비슷
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32 entries, 0 to 31
    Data columns (total 7 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Date         31 non-null     object 
     1   Location     32 non-null     object 
     2   Lemon        32 non-null     int64  
     3   Orange       32 non-null     int64  
     4   Temperature  32 non-null     int64  
     5   Leaflets     31 non-null     float64
     6   Price        32 non-null     float64
    dtypes: float64(2), int64(3), object(2)
    memory usage: 1.9+ KB
    
