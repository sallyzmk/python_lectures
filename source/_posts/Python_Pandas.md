---
# title: "Python Pandas"
date: '2022-06-29 01:00'
---

# 판다스
파이썬에서
- Index 인덱스 : 숫자, 문자, 날짜 등 다 OK / BUT 중복값 X
- Series 시리즈 : 인덱스 + 컬럼 1개
- DataFrame 데이터 프레임 : 인덱스 + 컬럼 2개
- 데이터 프레임.groupby : groupby 클래스가 됨.

## 클래스
- 클래스가 모이면 라이브러리(패키지)가 됨
- 응용할 때 에러가 나면 - 버전, type(클라스) 확인


## 라이브러리 불러오기


```python
import pandas as pd
import numpy as np
print("pandas version: ", pd.__version__)
print("numpy version: ", np.__version__)
```

    pandas version:  1.3.5
    numpy version:  1.21.6
    

## 데이터 불러오기
- 구글 드라이브에 데이터 존재


```python
# 구들 드라이브 연동
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
# 데이터 경로 지정
DATA_PATH ='/content/drive/MyDrive/Colab Notebooks/human_AI/Basic/Chapter 3. pandas/data/'
# 파일 불러오기
lemonade = pd.read_csv(DATA_PATH + 'Lemonade2016.csv')

# lemonade가 데이터프레임인지 시리즈인지 판단
print(type(lemonade)) 
"""
  [결과 해석]
  object: 문자, int: 정수, float: 실수
  int64 / int8 차이 : 부피공간 차이
"""

# 데이터셋에 존재하는 컬럼명과 컬럼별 결측치, 컬럼별 데이터타입을 확인
lemonade.info()
"""
  info는 데이터 프레임에서만 사용 가능
  info(): 데이터셋에 존재하는 컬럼명과 컬럼별 결측치, 컬럼별 데이터타입을 확인할 때 사용한다.
  R의 str() 함수와 유사하다.
"""
```

    <class 'pandas.core.frame.DataFrame'>
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
    

### 데이터 맛보기
- 데이터.head(): 상위 5개 보여줌, 숫자 지정가능
- 데이터.tail(): 하위 5개 보여줌, 숫자 지정가능
- 데이터.describe(): 기술통계량 보는 함수
- 데이터.value_counts(): 범주형 데이터 빈도수 구하기
  + 시리즈에 사용 가능, 데이터 프레임에는 사용 불가
  + 여러 컬럼(=피처)의 데이터 빈도수를 구하고 싶으면 **반복문 사용**
- 블로그에 올릴 때는 print(함수)


```python
# 블로그 올릴때는
print(lemonade.head()) 

# 상위 5개 보여줌, 숫자 지정가능
lemonade.head()
```

           Date Location  Lemon  Orange  Temperature  Leaflets  Price
    0  7/1/2016     Park     97      67           70      90.0   0.25
    1  7/2/2016     Park     98      67           72      90.0   0.25
    2  7/3/2016     Park    110      77           71     104.0   0.25
    3  7/4/2016    Beach    134      99           76      98.0   0.25
    4  7/5/2016    Beach    159     118           78     135.0   0.25
    


```python
# 하위 5개 보여줌, 숫자 지정가능
print(lemonade.tail())
```

             Date Location  Lemon  Orange  Temperature  Leaflets  Price
    27  7/27/2016     Park    104      68           80      99.0   0.35
    28  7/28/2016     Park     96      63           82      90.0   0.35
    29  7/29/2016     Park    100      66           81      95.0   0.35
    30  7/30/2016    Beach     88      57           82      81.0   0.35
    31  7/31/2016    Beach     76      47           82      68.0   0.35
    

- 데이터.describe(): 기술통계량 보는 함수


```python
# 기술통계량을 보여줌
print(lemonade.describe())
"""
  [결과 해석]
  lemon(25.823357)이 orange(21.863211) 보다 표준편차가 더 크다. 
  오렌지 판매량이 더 안정적이라는 의미. 
"""
```

                Lemon      Orange  Temperature    Leaflets      Price
    count   32.000000   32.000000    32.000000   31.000000  32.000000
    mean   116.156250   80.000000    78.968750  108.548387   0.354688
    std     25.823357   21.863211     4.067847   20.117718   0.113137
    min     71.000000   42.000000    70.000000   68.000000   0.250000
    25%     98.000000   66.750000    77.000000   90.000000   0.250000
    50%    113.500000   76.500000    80.500000  108.000000   0.350000
    75%    131.750000   95.000000    82.000000  124.000000   0.500000
    max    176.000000  129.000000    84.000000  158.000000   0.500000
    


```python
print(lemonade.Price.head())
"""
  [결과 해석]
  원가는 비슷비슷하기 때문에 평균편차가 크지 않음.
"""
```

    27    0.35
    28    0.35
    29    0.35
    30    0.35
    31    0.35
    Name: Price, dtype: float64
    

- 데이터.value_counts(): 범주형 데이터 빈도수 구하기
  + 시리즈에 사용 가능, 데이터 프레임에는 사용 불가
  + 여러 컬럼(=피처)의 데이터 빈도수를 구하고 싶으면 **반복문 사용**


```python
print(type(lemonade["Location"]))
lemonade["Location"]
```




    0      Park
    1      Park
    2      Park
    3     Beach
    4     Beach
    5     Beach
    6     Beach
    7     Beach
    8     Beach
    9     Beach
    10    Beach
    11    Beach
    12    Beach
    13    Beach
    14    Beach
    15    Beach
    16    Beach
    17    Beach
    18     Park
    19     Park
    20     Park
    21     Park
    22     Park
    23     Park
    24     Park
    25     Park
    26     Park
    27     Park
    28     Park
    29     Park
    30    Beach
    31    Beach
    Name: Location, dtype: object




```python
lemonade["Location"].value_counts()
```




    Beach    17
    Park     15
    Name: Location, dtype: int64



## 행과 열 다루기
- Sold(판매량) 컬럼(=피처, feature)을 추가


```python
# lemonade['새로운 변수']
lemonade['Sold']= 0
print(lemonade.head(3))
```

           Date Location  Lemon  Orange  Temperature  Leaflets  Price  Sold
    0  7/1/2016     Park     97      67           70      90.0   0.25     0
    1  7/2/2016     Park     98      67           72      90.0   0.25     0
    2  7/3/2016     Park    110      77           71     104.0   0.25     0
    


```python
# 전체 판매량 구하기 -> 칼럼끼리 더하면 된다.
lemonade['Sold'] = lemonade['Lemon'] + lemonade['Orange']
print(lemonade.head(3))
```

           Date Location  Lemon  Orange  Temperature  Leaflets  Price  Sold
    0  7/1/2016     Park     97      67           70      90.0   0.25   164
    1  7/2/2016     Park     98      67           72      90.0   0.25   165
    2  7/3/2016     Park    110      77           71     104.0   0.25   187
    

- Revenue(매출) = 단가 * 판매량


```python
lemonade['Revenue'] = lemonade['Price'] * lemonade['Sold']
print(lemonade.head())
```

           Date Location  Lemon  Orange  Temperature  Leaflets  Price  Sold  \
    0  7/1/2016     Park     97      67           70      90.0   0.25   164   
    1  7/2/2016     Park     98      67           72      90.0   0.25   165   
    2  7/3/2016     Park    110      77           71     104.0   0.25   187   
    3  7/4/2016    Beach    134      99           76      98.0   0.25   233   
    4  7/5/2016    Beach    159     118           78     135.0   0.25   277   
    
       Revenue  
    0    41.00  
    1    41.25  
    2    46.75  
    3    58.25  
    4    69.25  
    0    41.00
    1    41.25
    2    46.75
    3    58.25
    4    69.25
    Name: Revenue, dtype: float64
    


```python
print(lemonade[['Revenue', 'Price', 'Sold']].head())
```

       Revenue  Price  Sold
    0    41.00   0.25   164
    1    41.25   0.25   165
    2    46.75   0.25   187
    3    58.25   0.25   233
    4    69.25   0.25   277
    

- drop() 함수 사용해서 열 제거, 행 제거
- axis = 1 : columns 컬럼이 제거되는 것 -> 열 제거
- axis = 0 : 인덱스 번호에 해당되는 행이 제거되는 것 -> 행제거


```python
# 컬럼 제거
col_drop = lemonade.drop('Sold', axis =1)
print(col_drop.head())
```

           Date Location  Lemon  Orange  Temperature  Leaflets  Price  Revenue
    0  7/1/2016     Park     97      67           70      90.0   0.25    41.00
    1  7/2/2016     Park     98      67           72      90.0   0.25    41.25
    2  7/3/2016     Park    110      77           71     104.0   0.25    46.75
    3  7/4/2016    Beach    134      99           76      98.0   0.25    58.25
    4  7/5/2016    Beach    159     118           78     135.0   0.25    69.25
    


```python
# 행제거
# row_drop =lemonade.drop(인덱스 번호, axis=0)
row_drop =lemonade.drop(2, axis=0)
# 여러개 제거 할 때는 리스트로 제거 가능
# row_drop =lemonade.drop([0, 1, 2], axis=0)
print(row_drop.head())
```

           Date Location  Lemon  Orange  Temperature  Leaflets  Price  Sold  \
    0  7/1/2016     Park     97      67           70      90.0   0.25   164   
    1  7/2/2016     Park     98      67           72      90.0   0.25   165   
    3  7/4/2016    Beach    134      99           76      98.0   0.25   233   
    4  7/5/2016    Beach    159     118           78     135.0   0.25   277   
    5  7/6/2016    Beach    103      69           82      90.0   0.25   172   
    
       Revenue  
    0    41.00  
    1    41.25  
    3    58.25  
    4    69.25  
    5    43.00  
    

## 데이터 인덱싱


```python
print(lemonade[4:7])
```

           Date Location  Lemon  Orange  Temperature  Leaflets  Price  Sold  \
    4  7/5/2016    Beach    159     118           78     135.0   0.25   277   
    5  7/6/2016    Beach    103      69           82      90.0   0.25   172   
    6  7/6/2016    Beach    103      69           82      90.0   0.25   172   
    
       Revenue  
    4    69.25  
    5    43.00  
    6    43.00  
    

## 특정 값만 추출
- filter X
- 조건식 사용 : [ ] 안의 내용이 조건식
- 조건식 1개
- 데이터[데이터['데이터 컬럼'] == 특정 값]
- lemonade[lemonade['Location']  == 'Beach']
- 
- 조건식 2개
- & and, | or
- 데이터[(데이터['데이터 컬럼'] 조건식 ) & (데이터['데이터 컬럼'] 조건식 )]
- lemonade[(lemonade['Lemon'] >= 80) & (lemonade['Orange'] >= 100)]


```python
# 데이터[조건식_ 데이터['데이터 컬럼'] == 특정 값]
lemonade[lemonade['Location']  == 'Beach']
# 조건식
# 데이터['데이터 컬럼'] == 특정 값   -->> T, F
# T인 애들만 추출 됨.
```





  <div id="df-cf3d290c-3ecc-4bbe-9b6a-5fff5504efdd">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>Lemon</th>
      <th>Orange</th>
      <th>Temperature</th>
      <th>Leaflets</th>
      <th>Price</th>
      <th>Sold</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>7/4/2016</td>
      <td>Beach</td>
      <td>134</td>
      <td>99</td>
      <td>76</td>
      <td>98.0</td>
      <td>0.25</td>
      <td>233</td>
      <td>58.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7/5/2016</td>
      <td>Beach</td>
      <td>159</td>
      <td>118</td>
      <td>78</td>
      <td>135.0</td>
      <td>0.25</td>
      <td>277</td>
      <td>69.25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7/6/2016</td>
      <td>Beach</td>
      <td>103</td>
      <td>69</td>
      <td>82</td>
      <td>90.0</td>
      <td>0.25</td>
      <td>172</td>
      <td>43.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7/6/2016</td>
      <td>Beach</td>
      <td>103</td>
      <td>69</td>
      <td>82</td>
      <td>90.0</td>
      <td>0.25</td>
      <td>172</td>
      <td>43.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7/7/2016</td>
      <td>Beach</td>
      <td>143</td>
      <td>101</td>
      <td>81</td>
      <td>135.0</td>
      <td>0.25</td>
      <td>244</td>
      <td>61.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>Beach</td>
      <td>123</td>
      <td>86</td>
      <td>82</td>
      <td>113.0</td>
      <td>0.25</td>
      <td>209</td>
      <td>52.25</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7/9/2016</td>
      <td>Beach</td>
      <td>134</td>
      <td>95</td>
      <td>80</td>
      <td>126.0</td>
      <td>0.25</td>
      <td>229</td>
      <td>57.25</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7/10/2016</td>
      <td>Beach</td>
      <td>140</td>
      <td>98</td>
      <td>82</td>
      <td>131.0</td>
      <td>0.25</td>
      <td>238</td>
      <td>59.50</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7/11/2016</td>
      <td>Beach</td>
      <td>162</td>
      <td>120</td>
      <td>83</td>
      <td>135.0</td>
      <td>0.25</td>
      <td>282</td>
      <td>70.50</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7/12/2016</td>
      <td>Beach</td>
      <td>130</td>
      <td>95</td>
      <td>84</td>
      <td>99.0</td>
      <td>0.25</td>
      <td>225</td>
      <td>56.25</td>
    </tr>
    <tr>
      <th>13</th>
      <td>7/13/2016</td>
      <td>Beach</td>
      <td>109</td>
      <td>75</td>
      <td>77</td>
      <td>99.0</td>
      <td>0.25</td>
      <td>184</td>
      <td>46.00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>7/14/2016</td>
      <td>Beach</td>
      <td>122</td>
      <td>85</td>
      <td>78</td>
      <td>113.0</td>
      <td>0.25</td>
      <td>207</td>
      <td>51.75</td>
    </tr>
    <tr>
      <th>15</th>
      <td>7/15/2016</td>
      <td>Beach</td>
      <td>98</td>
      <td>62</td>
      <td>75</td>
      <td>108.0</td>
      <td>0.50</td>
      <td>160</td>
      <td>80.00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>7/16/2016</td>
      <td>Beach</td>
      <td>81</td>
      <td>50</td>
      <td>74</td>
      <td>90.0</td>
      <td>0.50</td>
      <td>131</td>
      <td>65.50</td>
    </tr>
    <tr>
      <th>17</th>
      <td>7/17/2016</td>
      <td>Beach</td>
      <td>115</td>
      <td>76</td>
      <td>77</td>
      <td>126.0</td>
      <td>0.50</td>
      <td>191</td>
      <td>95.50</td>
    </tr>
    <tr>
      <th>30</th>
      <td>7/30/2016</td>
      <td>Beach</td>
      <td>88</td>
      <td>57</td>
      <td>82</td>
      <td>81.0</td>
      <td>0.35</td>
      <td>145</td>
      <td>50.75</td>
    </tr>
    <tr>
      <th>31</th>
      <td>7/31/2016</td>
      <td>Beach</td>
      <td>76</td>
      <td>47</td>
      <td>82</td>
      <td>68.0</td>
      <td>0.35</td>
      <td>123</td>
      <td>43.05</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cf3d290c-3ecc-4bbe-9b6a-5fff5504efdd')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-cf3d290c-3ecc-4bbe-9b6a-5fff5504efdd button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cf3d290c-3ecc-4bbe-9b6a-5fff5504efdd');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# 조건식
# 데이터['데이터 컬럼'] == 특정 값   -->> T, F
# T인 애들만 추출 됨.
lemonade['Location']  == 'Beach'
```




    0     False
    1     False
    2     False
    3      True
    4      True
    5      True
    6      True
    7      True
    8      True
    9      True
    10     True
    11     True
    12     True
    13     True
    14     True
    15     True
    16     True
    17     True
    18    False
    19    False
    20    False
    21    False
    22    False
    23    False
    24    False
    25    False
    26    False
    27    False
    28    False
    29    False
    30     True
    31     True
    Name: Location, dtype: bool




```python
lemonade[lemonade['Temperature'] >= 80]
```





  <div id="df-c9af1370-d710-4f63-91ac-f43d9b1234a0">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>Lemon</th>
      <th>Orange</th>
      <th>Temperature</th>
      <th>Leaflets</th>
      <th>Price</th>
      <th>Sold</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>7/6/2016</td>
      <td>Beach</td>
      <td>103</td>
      <td>69</td>
      <td>82</td>
      <td>90.0</td>
      <td>0.25</td>
      <td>172</td>
      <td>43.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7/6/2016</td>
      <td>Beach</td>
      <td>103</td>
      <td>69</td>
      <td>82</td>
      <td>90.0</td>
      <td>0.25</td>
      <td>172</td>
      <td>43.00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7/7/2016</td>
      <td>Beach</td>
      <td>143</td>
      <td>101</td>
      <td>81</td>
      <td>135.0</td>
      <td>0.25</td>
      <td>244</td>
      <td>61.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>Beach</td>
      <td>123</td>
      <td>86</td>
      <td>82</td>
      <td>113.0</td>
      <td>0.25</td>
      <td>209</td>
      <td>52.25</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7/9/2016</td>
      <td>Beach</td>
      <td>134</td>
      <td>95</td>
      <td>80</td>
      <td>126.0</td>
      <td>0.25</td>
      <td>229</td>
      <td>57.25</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7/10/2016</td>
      <td>Beach</td>
      <td>140</td>
      <td>98</td>
      <td>82</td>
      <td>131.0</td>
      <td>0.25</td>
      <td>238</td>
      <td>59.50</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7/11/2016</td>
      <td>Beach</td>
      <td>162</td>
      <td>120</td>
      <td>83</td>
      <td>135.0</td>
      <td>0.25</td>
      <td>282</td>
      <td>70.50</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7/12/2016</td>
      <td>Beach</td>
      <td>130</td>
      <td>95</td>
      <td>84</td>
      <td>99.0</td>
      <td>0.25</td>
      <td>225</td>
      <td>56.25</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7/18/2016</td>
      <td>Park</td>
      <td>131</td>
      <td>92</td>
      <td>81</td>
      <td>122.0</td>
      <td>0.50</td>
      <td>223</td>
      <td>111.50</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7/22/2016</td>
      <td>Park</td>
      <td>112</td>
      <td>75</td>
      <td>80</td>
      <td>108.0</td>
      <td>0.50</td>
      <td>187</td>
      <td>93.50</td>
    </tr>
    <tr>
      <th>23</th>
      <td>7/23/2016</td>
      <td>Park</td>
      <td>120</td>
      <td>82</td>
      <td>81</td>
      <td>117.0</td>
      <td>0.50</td>
      <td>202</td>
      <td>101.00</td>
    </tr>
    <tr>
      <th>24</th>
      <td>7/24/2016</td>
      <td>Park</td>
      <td>121</td>
      <td>82</td>
      <td>82</td>
      <td>117.0</td>
      <td>0.50</td>
      <td>203</td>
      <td>101.50</td>
    </tr>
    <tr>
      <th>25</th>
      <td>7/25/2016</td>
      <td>Park</td>
      <td>156</td>
      <td>113</td>
      <td>84</td>
      <td>135.0</td>
      <td>0.50</td>
      <td>269</td>
      <td>134.50</td>
    </tr>
    <tr>
      <th>26</th>
      <td>7/26/2016</td>
      <td>Park</td>
      <td>176</td>
      <td>129</td>
      <td>83</td>
      <td>158.0</td>
      <td>0.35</td>
      <td>305</td>
      <td>106.75</td>
    </tr>
    <tr>
      <th>27</th>
      <td>7/27/2016</td>
      <td>Park</td>
      <td>104</td>
      <td>68</td>
      <td>80</td>
      <td>99.0</td>
      <td>0.35</td>
      <td>172</td>
      <td>60.20</td>
    </tr>
    <tr>
      <th>28</th>
      <td>7/28/2016</td>
      <td>Park</td>
      <td>96</td>
      <td>63</td>
      <td>82</td>
      <td>90.0</td>
      <td>0.35</td>
      <td>159</td>
      <td>55.65</td>
    </tr>
    <tr>
      <th>29</th>
      <td>7/29/2016</td>
      <td>Park</td>
      <td>100</td>
      <td>66</td>
      <td>81</td>
      <td>95.0</td>
      <td>0.35</td>
      <td>166</td>
      <td>58.10</td>
    </tr>
    <tr>
      <th>30</th>
      <td>7/30/2016</td>
      <td>Beach</td>
      <td>88</td>
      <td>57</td>
      <td>82</td>
      <td>81.0</td>
      <td>0.35</td>
      <td>145</td>
      <td>50.75</td>
    </tr>
    <tr>
      <th>31</th>
      <td>7/31/2016</td>
      <td>Beach</td>
      <td>76</td>
      <td>47</td>
      <td>82</td>
      <td>68.0</td>
      <td>0.35</td>
      <td>123</td>
      <td>43.05</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c9af1370-d710-4f63-91ac-f43d9b1234a0')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c9af1370-d710-4f63-91ac-f43d9b1234a0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c9af1370-d710-4f63-91ac-f43d9b1234a0');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
lemonade[(lemonade['Lemon'] >= 80) & (lemonade['Orange'] >= 100)]
```





  <div id="df-5e3874ab-d1a0-45aa-92b9-97df1dfe1239">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>Lemon</th>
      <th>Orange</th>
      <th>Temperature</th>
      <th>Leaflets</th>
      <th>Price</th>
      <th>Sold</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>7/5/2016</td>
      <td>Beach</td>
      <td>159</td>
      <td>118</td>
      <td>78</td>
      <td>135.0</td>
      <td>0.25</td>
      <td>277</td>
      <td>69.25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7/7/2016</td>
      <td>Beach</td>
      <td>143</td>
      <td>101</td>
      <td>81</td>
      <td>135.0</td>
      <td>0.25</td>
      <td>244</td>
      <td>61.00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7/11/2016</td>
      <td>Beach</td>
      <td>162</td>
      <td>120</td>
      <td>83</td>
      <td>135.0</td>
      <td>0.25</td>
      <td>282</td>
      <td>70.50</td>
    </tr>
    <tr>
      <th>25</th>
      <td>7/25/2016</td>
      <td>Park</td>
      <td>156</td>
      <td>113</td>
      <td>84</td>
      <td>135.0</td>
      <td>0.50</td>
      <td>269</td>
      <td>134.50</td>
    </tr>
    <tr>
      <th>26</th>
      <td>7/26/2016</td>
      <td>Park</td>
      <td>176</td>
      <td>129</td>
      <td>83</td>
      <td>158.0</td>
      <td>0.35</td>
      <td>305</td>
      <td>106.75</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5e3874ab-d1a0-45aa-92b9-97df1dfe1239')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-5e3874ab-d1a0-45aa-92b9-97df1dfe1239 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5e3874ab-d1a0-45aa-92b9-97df1dfe1239');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
lemonade[(lemonade['Lemon'] >= 80) & (lemonade['Orange'] >= 100) & (lemonade['Location'] == 'Beach')]
```





  <div id="df-a27156bb-3624-45d1-a577-390139ddcf4e">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
      <th>Lemon</th>
      <th>Orange</th>
      <th>Temperature</th>
      <th>Leaflets</th>
      <th>Price</th>
      <th>Sold</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>7/5/2016</td>
      <td>Beach</td>
      <td>159</td>
      <td>118</td>
      <td>78</td>
      <td>135.0</td>
      <td>0.25</td>
      <td>277</td>
      <td>69.25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7/7/2016</td>
      <td>Beach</td>
      <td>143</td>
      <td>101</td>
      <td>81</td>
      <td>135.0</td>
      <td>0.25</td>
      <td>244</td>
      <td>61.00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7/11/2016</td>
      <td>Beach</td>
      <td>162</td>
      <td>120</td>
      <td>83</td>
      <td>135.0</td>
      <td>0.25</td>
      <td>282</td>
      <td>70.50</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a27156bb-3624-45d1-a577-390139ddcf4e')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-a27156bb-3624-45d1-a577-390139ddcf4e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a27156bb-3624-45d1-a577-390139ddcf4e');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## 특정 칼럼만 보고 싶을 때
- iloc : 인덱스 기반, 숫자 기반, [4, 3, 2]
- loc : 칼람명 기반, 문자 기반, ['a', 'b', 'c']


```python
# 특정 칼럼만 보고 싶을 때
lemonade.loc[lemonade['Temperature'] >= 80, ['Date', 'Location']]
```





  <div id="df-e0559f11-b1eb-4d94-81cf-2b0dca9703ad">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Location</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>7/6/2016</td>
      <td>Beach</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7/6/2016</td>
      <td>Beach</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7/7/2016</td>
      <td>Beach</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>Beach</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7/9/2016</td>
      <td>Beach</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7/10/2016</td>
      <td>Beach</td>
    </tr>
    <tr>
      <th>11</th>
      <td>7/11/2016</td>
      <td>Beach</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7/12/2016</td>
      <td>Beach</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7/18/2016</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7/22/2016</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>23</th>
      <td>7/23/2016</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>24</th>
      <td>7/24/2016</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>25</th>
      <td>7/25/2016</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>26</th>
      <td>7/26/2016</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>27</th>
      <td>7/27/2016</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>28</th>
      <td>7/28/2016</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>29</th>
      <td>7/29/2016</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>30</th>
      <td>7/30/2016</td>
      <td>Beach</td>
    </tr>
    <tr>
      <th>31</th>
      <td>7/31/2016</td>
      <td>Beach</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e0559f11-b1eb-4d94-81cf-2b0dca9703ad')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e0559f11-b1eb-4d94-81cf-2b0dca9703ad button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e0559f11-b1eb-4d94-81cf-2b0dca9703ad');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## 일부 데이터만 추출 (iloc vs loc)
- 문법상의 차이 확인
- iloc : 숫자만 들어감, 필터링이 안먹힘
- loc : 라벨 =글자 (숫자+문자), 필터링이 된다.
- loc 사용 선호


```python
lemonade.head()

# 일부 데이터만 추출 iloc
# print(lemonade.iloc[행, 열])
print(lemonade.iloc[0:3, 0:2])
```

           Date Location
    0  7/1/2016     Park
    1  7/2/2016     Park
    2  7/3/2016     Park
    


```python
# 일부 데이터만 추출 loc
# print(lemonade.loc[인덱스X 첫 행에 있는 숫자이다,['칼럼명', '칼람명']])
lemonade.head()
print(lemonade.loc[0:2,['Date', 'Location']])
```

           Date Location
    0  7/1/2016     Park
    1  7/2/2016     Park
    2  7/3/2016     Park
    

## 데이터 정렬
- sort_values


```python
lemonade.head()
print(lemonade.sort_values(by=['Revenue']).head(10))
```

             Date Location  Lemon  Orange  Temperature  Leaflets  Price  Sold  \
    0    7/1/2016     Park     97      67           70      90.0   0.25   164   
    1    7/2/2016     Park     98      67           72      90.0   0.25   165   
    6    7/6/2016    Beach    103      69           82      90.0   0.25   172   
    5    7/6/2016    Beach    103      69           82      90.0   0.25   172   
    31  7/31/2016    Beach     76      47           82      68.0   0.35   123   
    13  7/13/2016    Beach    109      75           77      99.0   0.25   184   
    2    7/3/2016     Park    110      77           71     104.0   0.25   187   
    30  7/30/2016    Beach     88      57           82      81.0   0.35   145   
    14  7/14/2016    Beach    122      85           78     113.0   0.25   207   
    8         NaN    Beach    123      86           82     113.0   0.25   209   
    
        Revenue  
    0     41.00  
    1     41.25  
    6     43.00  
    5     43.00  
    31    43.05  
    13    46.00  
    2     46.75  
    30    50.75  
    14    51.75  
    8     52.25  
    


```python
print(lemonade[['Date', 'Temperature','Revenue']].sort_values(by=['Revenue']).head(5))
```

             Date  Temperature  Revenue
    0    7/1/2016           70    41.00
    1    7/2/2016           72    41.25
    6    7/6/2016           82    43.00
    5    7/6/2016           82    43.00
    31  7/31/2016           82    43.05
    


```python
"""
  데이터.sort_values(by= ['컬럼1', '컬럼2'])
  컬럼 1 먼저 내림차순으로 정리 후, 컬럼1의 값이 동일한 열끼리 컬럼 2 기준으로 내림차순 정리.
"""
print(lemonade[['Date', 'Temperature','Revenue']].sort_values(by=['Temperature','Revenue']).head(5))
```

             Date  Temperature  Revenue
    0    7/1/2016           70    41.00
    20  7/20/2016           70    56.50
    2    7/3/2016           71    46.75
    1    7/2/2016           72    41.25
    16  7/16/2016           74    65.50
    


```python
"""
  1. 내림차순 내림차순
  데이터.sort_values(by=['컬럼1','컬럼2'],ascending = [True, True])

  2. 내림차순 오름차순
  데이터.sort_values(by=['컬럼1','컬럼2'],ascending = [True, False])

  3. 오름차순 내림차순
  데이터.sort_values(by=['컬럼1','컬럼2'],ascending = [False, True])

  4. 오름차순 오름차순
  데이터.sort_values(by=['컬럼1','컬럼2'],ascending = [False, False])

  5. 컬럼 수가 늘면 T,F도 숫자를 맞춰춰야함
    데이터.sort_values(by=['컬럼1','컬럼2','컬럼3'],ascending = [False, False, True])
"""
print(lemonade[['Date', 'Temperature','Revenue']].sort_values(by=['Temperature','Revenue'],ascending = [True, False]).head(5))
```

             Date  Temperature  Revenue
    20  7/20/2016           70    56.50
    0    7/1/2016           70    41.00
    2    7/3/2016           71    46.75
    1    7/2/2016           72    41.25
    16  7/16/2016           74    65.50
    

## Group by


```python
df = lemonade.groupby(by='Location').count()
print(df)
print(type(df))
"""
  [결과 분석]
  - 숫자가 적은 것은 결측치
  - DataFrame : 형태가 약간 다르나, 인덱스가 문자인 것 뿐이다.
"""
```

              Date  Lemon  Orange  Temperature  Leaflets  Price  Sold  Revenue
    Location                                                                  
    Beach       16     17      17           17        17     17    17       17
    Park        15     15      15           15        14     15    15       15
    <class 'pandas.core.frame.DataFrame'>
    




    '\n  [결과 분석]\n  - 숫자가 적은 것은 결측치\n  - DataFrame : 형태가 약간 다르나, 인덱스가 문자인 것 뿐이다.\n'




```python
df[['Date', 'Lemon']]
```


```python
print(df.iloc[0:1, 0:2])
```


```python
print(df.loc['Park', ['Date', 'Lemon']])
```

## 간단한 피벗 테이블 만들기


```python
# lemonade.groupby('범주형 데이터 컬럼명')['컬럼명'].agg([기술 통계량 리스트])
lemonade.groupby('Location')['Revenue'].agg([max, min, sum, np.mean])
"""
  lemonade 데이터를 Location지역을 기준으로 groupby 표를 만든다.
  근데 전체 데이터를 표로 만드는 것이 아닌, 특정 칼럼(Revenue수익)의 기술통계량을 알아보고자 한다.
"""
```




    '\n  lemonade 데이터를 Location지역을 기준으로 groupby 표를 만든다.\n  근데 전체 데이터를 표로 만드는 것이 아닌, 특정 칼럼(Revenue수익)의 기술통계량을 알아보고자 한다.\n'






```python
# # lemonade.groupby('범주형 데이터 컬럼명')[['컬럼명1', '컬럼명2', '컬럼명3'-->> 컬럼명 리스트]].agg([기술 통계량 리스트])
lemonade.groupby('Location')[['Revenue','Sold']].agg([max, min, sum, np.mean])
"""
  lemonade 데이터를 Location지역을 기준으로 groupby 표를 만든다.
  근데 전체 데이터를 표로 만드는 것이 아닌, 특정 칼럼(Revenue수익, Sold판매총액)의 기술통계량을 알아보고자 한다.
"""

p
```




    '\n  lemonade 데이터를 Location지역을 기준으로 groupby 표를 만든다.\n  근데 전체 데이터를 표로 만드는 것이 아닌, 특정 칼럼(Revenue수익, Sold판매총액)의 기술통계량을 알아보고자 한다.\n'




```python
# lemonade.groupby(['범주형 데이터 컬럼명1','범주형 데이터 컬럼명2'-->>리스트로 만들 수 있음])['컬럼명'].agg([기술 통계량 리스트])
lemonade.groupby(['Location', 'Price'])['Orange'].agg([max, min, sum, np.mean])
"""
  lemonade 데이터를 Location지역과 Price가격을 기준으로 groupby 표를 만든다.
  근데 전체 데이터를 표로 만드는 것이 아닌, 특정 칼럼(Orange판매된 레몬 갯수)의 기술통계량을 알아보고자 한다.
  Beach에서 0.25에 오렌지를 판매 했을 때 최대 120개, 최소 69개 판매했다.
"""
```




    '\n  lemonade 데이터를 Location지역과 Price가격을 기준으로 groupby 표를 만든다.\n  근데 전체 데이터를 표로 만드는 것이 아닌, 특정 칼럼(Orange판매된 레몬 갯수)의 기술통계량을 알아보고자 한다.\n  Beach에서 0.25에 오렌지를 판매 했을 때 최대 120개, 최소 69개 판매했다.\n'


