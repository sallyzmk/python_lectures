---
# title: "Python 기초문법"
date: '2022-06-27 02:00'
---

## 자율학습
- [점프 투 파이썬](https://wikidocs.net/book/1)
- [파이썬 코딩 도장](https://dojang.io/course/view.php?id=7)

- 파이썬을 실행하는 도구
  + 파이참 : 윈도우
  + visual studio code : 리눅스
  + 주피터, 코랩 : 강의, 블로그

- 데이터 과학과 관련된 주요 라이브러리
  + 구글 코랩 : 이미 설치 됨
  + 로컬 : 환경설정을 추가로 해야함 (=가상환경)

- 문자열, 리스트는 내장함수를 익히는 것이 포인트

## Hello World 출력




```python
print("Hello World")
```

    Hello World
    

## 주석처리
- 1줄 주석 (#)
- 여러 줄 주석 처리 ("""  """)
  + 함수 또는 클래스를 문서화 할 때 주로 사용

- 프로젝트 할 때
  + 전체 공정 100
  + 코드 / 코드 문서화 / 한글작업 문서화


```python
# 1줄 주석
print("1줄 주석")

"""
여러 줄 주석
쌍따옴표 3개를 수미에 입력해주세요
"""
print("여러줄 주석")
```

    1줄 주석
    여러줄 주석
    

## 변수 (Scalar)
- 자료형 - Scalar형, Non-Scalar형

- Scalar형
1. 파이썬 정수 변환 - int()
2. 파이썬 실수 변환 - float()
5. 파이썬 불리언 변환 - bool()
6. None 자료형 - Null값, 값이 정해지지 않은 자료형

- Non-Scalar형
3. 파이썬 문자열 변환 - str()
*4. 파이썬 문자 변환 - chr()*
- https://blockdmask.tistory.com/432

### 수치형 자료형 (int, float)
- int(정수): 정수를 정수 타입으로 변환하여 반환
- int(문자열): 문자열에 맞는 정수 타입 반환, 반환 불가 문자열-> 오류남.
- int(불리언): True =1, False = 0


```python
num_int = 1
print(num_int)
print(type(num_int))
```

    1
    <class 'int'>
    


```python
num_float = 0.1
print(num_float)
print(type(num_float))
```

    0.1
    <class 'float'>
    

### Bool형 (True, False)
- Python : True, False
- R : TRUE, FALSE


```python
bool_true = True
print(bool_true)
print(type(bool_true))
```

    True
    <class 'bool'>
    

### None 자료형
- Null값, 값이 정해지지 않은 자료형


```python
none_X = None
print(none_X)
print(type(none_X))
```

    None
    <class 'NoneType'>
    

## 사칙연산
- 정수형 사칙연산, 실수형 사칙연산
- !!!결괏값의 자료형 확인하기!!! 중요!
- 因 정수형으로 계산했으나 결과가 실수형이 나올 수 있음

### 정수형 사칙연산
- +, -, *, /,
- **: 제곱
- % : 나눗셈 후 나머지를 반환
- //: 나눗셈 후 몫을 반환


```python
a = 7
b = 3
print('a + b = ', a+b)
print('a - b = ', a-b)
print('a * b = ', a*b)
print('a / b = ', a/b)
print('a ** b = ', a**b)
print('a % b = ', a%b)
print('a // b = ', a//b)
```

    a + b =  10
    a - b =  4
    a * b =  21
    a / b =  2.3333333333333335
    a ** b =  343
    a % b =  1
    a // b =  2
    

### 실수형 사칙 연산
+, -, *, /


```python
a = 1.5
b = 2.5
print('a + b = ', a+b)
print('a - b = ', a-b)
print('a * b = ', a*b)
print('a / b = ', a/b)
```

    a + b =  4.0
    a - b =  -1.0
    a * b =  3.75
    a / b =  0.6
    hello
    

## 논리형 연산자
- Bool형
- True와 False 값으로 정의
- 조건식
  + 교집합(=and), 합집합(=or)


```python
print(True and True)
print(True and False)
print(False and True)
print(False and False)
```

    True
    False
    False
    False
    


```python
print(True or True)
print(True or False)
print(False or True)
print(False or False)
```

    True
    True
    True
    False
    

## 비교 연산자 (>, <)
- 비교 연산자는 부등호를 의미한다.


```python
print(4 > 3) # 참 = True
print(4 < 3) # 거짓 = False

# 응용 (논리형 & 비교 연산자 응용)
print(4 > 3 and 4 < 3)
print(4 > 3 or 4 < 3)
```

    True
    False
    False
    True
    

## 논리형 & 비교 연산자 응용
- input()
- 형변환 : 데이터 타입을 바꾸는 것


```python
# var = input("숫자를 입력하세요..!") # str 문자열
var = int(input("숫자를 입력하세요..!")) # int 숫자열
print(var)
print(type(var))
```

    숫자를 입력하세요..!1
    1
    <class 'int'>
    


```python
num1 = i]nt(input("첫번째 숫자를 입력하세요..!"))
num2 = int(input("두번째 숫자를 입력하세요..!"))
num3 = int(input("세번째 숫자를 입력하세요..!"))
num4 = int(input("네번째 숫자를 입력하세요..!"))
```

    첫번째 숫자를 입력하세요..!100
    두번째 숫자를 입력하세요..!22
    세번째 숫자를 입력하세요..!46
    네번째 숫자를 입력하세요..!577
    


```python
var1 = num1 >= num2
var2 = num3 < num2
```


```python
print(var1 and var2)
print(var1 or var2)
```

    False
    True
    

## 문자열 (String) (Non Scalar)
- \n	문자열 안에서 줄을 바꿀 때 사용
- \t	문자열 사이에 탭 간격을 줄 때 사용
- \\	문자 \를 그대로 표현할 때 사용
- \'	작은따옴표(')를 그대로 표현할 때 사용
- \"	큰따옴표(")를 그대로 표현할 때 사용
- \r	캐리지 리턴(줄 바꿈 문자, 현재 커서를 가장 앞으로 이동)
- \f	폼 피드(줄 바꿈 문자, 현재 커서를 다음 줄로 이동)
- \a	벨 소리(출력할 때 PC 스피커에서 '삑' 소리가 난다)
- \b	백 스페이스
- \000	널 문자

- 이중에서 활용빈도가 높은 것은 \n, \t, \\, \', \"이다. 나머지는 프로그램에서 잘 사용하지 않는다.


```python
print('Hello World')
print("Hello World")

print("'Hello World'") # 작은 따옴표를 표시해야될 때
print('"Hello World"') # 큰 따옴표를 표시해야될 때

print('Hello World. \nMy name is Sally.') # 줄 바꿔야 할 때 방법1 (\n)

x = '''
Hello World.
My name is Sally.
'''  # 줄 바꿔야 할 때 방법2 (''' ''')
print(x)
```

    Hello World
    Hello World
    'Hello World'
    "Hello World"
    Hello World. 
    My name is Sally.
    
    Hello World.
    My name is Sally.
    
    

## String Operators (문자열 연산자)
- 문자열 연산자
- +, * 가능


```python
str1 = "Hello "
str2 = "World"
print(str1+str2)
```

    Hello World
    


```python
greet = str1 + str2
print(greet * 7) # n번 연속으로 출력된다.
```

    Hello WorldHello WorldHello WorldHello WorldHello WorldHello WorldHello World
    

## 인덱싱
- 인덱싱은 0번째 부터 시작
- 맨 마지막 열은 -1
- 빈칸도 한 개로 인식 함
- len() : 문자열 인덱싱 할 시에 길이를 알려주는 함수.


```python
greeting = "Hello Kaggle"
print(greeting)

print(greeting[0]) # 0 번째
print(greeting[-1]) # 마지막 번째
print(greeting[5]) # 빈칸

# 적용 예시
greeting = "Hello Kaggle"
i = int(input("숫자를 입력하세요...!"))
print(greeting[i])
```

    Hello Kaggle
    H
    e
    Hello
    숫자를 입력하세요...!7
    a
    

## 슬라이싱
- [시작 인덱스:끝 인데스-1]


```python
greeting = "Hello Kaggle"
(greeting)

print(greeting[:5]) # 0~4 번째
print(greeting[-5:]) # -5 ~ 끝 번째
print(greeting[-5:-1]) # -5 ~ -2 번째
print(greeting[:]) # 처음 ~ 끝 번째

print(greeting[0:10:2]) # 0 ~ 9 번째까지 2 스탭씩

```

    Hello
    aggle
    aggl
    Hello Kaggle
    HloKg
    

- 문자열 관련 메서드
  + split()
  + sort()
  + etc ...

## 리스트
- []로 표시
- [item1, item2, item3]
- 리스트에는 많은게 들어간다. 때문에 리스트 관리를 잘 해야 한다.


```python
a = [] # 빈 리스트 생성 방법 1
a_func = list() # 빈 리스트 생성 방법 2

b = [1] # 숫자 요소
c = ['apple'] # 문자요소
d = [1, 2, ['apple'], 'apple']
print(d)
```

    [1, 2, ['apple'], 'apple']
    

### 리스트 값 추가하기 (기본)


```python
a = [0, 1, 2]
a[1] = '아무값'
print(a)
```

    [0, '아무 값', 2]
    

### 리스트 값 추가하기 (메서드 사용)
- class에 기본 저장된 메서드


```python
a = [100, 200, 300]
a.append(400) # a = a.append(400) XXX, 메서드를 사용할 때에는 별도 변수 저장이 필요 없다.
print(a)

a.append([500, 600])  # [500, 600] 리스트로 들어가버림, 다른 메서드를 사용해야 함.
print(a)

a.extend([500, 600])
print(a)
```

    [100, 200, 300, 400]
    [100, 200, 300, 400, [500, 600]]
    [100, 200, 300, 400, [500, 600], 400, 500]
    

### 리스트 값 추가하기 (insert 사용)
- insert(인덱스 위치, 값)
- 리스트 맨 마지막에 값을 추가하는 것이 아닌 내가 **원하는 위치에 값을 추가**할 수 있다.


```python
a = [100, 200, 300]
a.insert(1, 1000)
print(a)
```

    [100, 1000, 200, 300]
    

### 리스트 값 삭제하기 (remove(), del)
- remove(): 특정 값 하나만 삭제, **같은 값이 2개 이상** 있을 경우 가장 **앞에 있는 값 하나만 삭제**됨.
- del 리스트[인덱스 번호]: 요소 번호, 인덱스 번호, 해당되는 요소가 삭제
- pop(): 
  + x = a.pop(1) # 인덱스 번호, 리스트의 2번째 요소를 삭제하라
  + print(x) : 삭제된 값을 나타냄. 리스트의 2번째 요소.
- clear(): 리스트 내 모든 값 삭제



```python
# remove()
a = [1, 2, 1, 2]
a.remove(2)
print(a)

b = [4, 5, 6, 7]
b.remove(6)
print(b)

```

    [1, 1, 2]
    [4, 5, 7]
    


```python
# del
a= [0, 1, 2, 3, 4]
del a[1] # 요소 번호, 인덱스 번호, 해당되는 요소가 삭제
print(a)

del a[0:2]
print(a)
```

    [0, 2, 3, 4]
    [3, 4]
    


```python
# pop()
a = [1, 2, 3, 4, 5]
a.pop(1) # 인덱스 번호
print(a)

b = [4, 5, 6, 7, 8]
rem = b.pop(2)
print(b)
print(rem)
```

    [1, 3, 4, 5]
    [4, 5, 7, 8]
    6
    


```python
a = [1, 2, 3, 4, 5]
rem = a.pop(1) # 인덱스 번호 -> 2번째 번호
print(a)
print(rem)
x = a.pop() # 인덱스 번호 -> 맨 끝 번호
print(a)
print(x)
```

    [1, 3, 4, 5]
    2
    [1, 3, 4]
    5
    

### 리스트 값의 위치 불러오기 (index)
- index("값"): 값의 위치를 불러옴


```python
# index() : 리스트 안의 값이 몇 번째 인덱스 순서에 해당되는지 알려줌
a = [1, 4, 5, 2, 3]
b = ["철수", "영희", "길동"]
print(a.index(4))
print(b.index("길동"))
```

### 리스트의 정렬 (sort)
- sort: 리스트의 정렬


```python
a = [1, 4, 5, 2, 3]
a.sort()
print(a)
b = [1, 4, 5, 2, 3]
b.sort(reverse=True)
print(b)


```

    [1, 2, 3, 4, 5]
    [5, 4, 3, 2, 1]
    

## 도움말 부르기
- help(list.sort)
- help(list.index)
- 등등...

## 튜플
- 면접질문 : 리스트와 튜플의 차이가 뭐에요?
- 1. 생김새가 다르다.
- 2. 수정, 삭제, 추가가 다 안된다.

- 리스트 : []
  + 수정, 삭제, 추가

- 튜플 : (,)
  + ', '찍어야 튜플로 인정, 안그럼 숫자로 인식.
  + 수정, 삭제, 추가가 다 안됨. 슬라이싱은 된다.
  + 바뀌어서는 안되는 리스트를 튜플을 사용.
  + 튜플>리스트>튜플 형변화로 수정은 가능하나..굳이? 그럴거면 그냥 리스트로 지정하세요!
  + 그냥 처음부터 지정을 잘 하자.


```python
tuple1 = (0)
tuple2 = (0, )
tuple3 = 0, 1, 2

print(type(tuple1))
print(type(tuple2))
print(type(tuple3))

print(tuple1)
print(tuple2)
print(tuple3)
```

    <class 'int'>
    <class 'tuple'>
    <class 'tuple'>
    0
    (0,)
    (0, 1, 2)
    


```python
a = (0, 1, 2, 3, 'a')
# del a[4]
# a[4] = 4

# doesn't support 지원하지 않는다.
# 튜플은 수정이 불가하다.
```

## 튜플, 리스트 연산자
- 문자열 연산자 (+, *)



```python
# 튜플
t1 = (0, 1, 2)
t2 = (3, 4, 5)
print(t1 + t2)
print(t1*2)

# 리스트
y1 = [0, 1, 2]
y2 = [3, 4, 5]
print(y1 + y2)
print(y1*2+y2*4)


```

    (0, 1, 2, 3, 4, 5)
    (0, 1, 2, 0, 1, 2)
    [0, 1, 2, 3, 4, 5]
    [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5]
    

## Dictionary
- Key와 VAlue(값)으로 구성됨
- 슬라이싱이 안된다. (슬라이싱은 값의 순서가 존재해야 함)
- 딕셔너리는 순서라는 개념 자체가 존재하지 않음


```python
temp_dict = {
    'teacher' : 'evan',
    'class' : 15,
    'students' : ['s1', 's2', 's3']
}

print(temp_dict['teacher'])
print(temp_dict['class'])
print(temp_dict['students'])
```

    evan
    15
    ['s1', 's2', 's3']
    

- keys() 값만 출력 (   list(temp_dict.keys())   )


```python
temp_dict.keys() # list는 아님.
list(temp_dict.keys()) # 형변환
```




    ['teacher', 'class', 'students']



- values() 값만 출력 (temp_dict.values())


```python
temp_dict.values()
```




    dict_values(['evan', 15, ['s1', 's2', 's3']])



- items(): key-value 쌍으로, list와 tuple 형태로 반환 


```python
temp_dict.items()
```




    dict_items([('teacher', 'evan'), ('class', 15), ('students', ['s1', 's2', 's3'])])



##  조건문 
- 잊지말고 ':' 찍기!!


```python
# a = -5
a = int(input("숫자를 입력하세요..!"))
if a > 5:
  print("a는 5보다 크다")
elif a > 0:
  print("a는 0보다 크다")
elif a > -5:
  print("a는 -5보다 크다")
else:
  print("a는 -5보다 작다")
```

    숫자를 입력하세요..!7
    a는 5보다 크다
    

## 반복문



```python
# Hello World n번 출력하세요

# 방법 1
for i in range(3):  # i는 변수, 바꿔도 된다.
  print(i+1)  # 파이썬은 0부터 시작
  print("Hello World")

# 방법 2
for i in range(3):  # i는 변수, 바꿔도 된다.
  print(i+1, "Hello World")
```

    1
    Hello World
    2
    Hello World
    3
    Hello World
    1 Hello World
    2 Hello World
    3 Hello World
    

- for loop if 조건문 사용 
- 문자열, 리스트 등 -> 시퀀스 데이터 (반복문 사용 가능)


```python
a = "Kaggle"
# g가 시작하면 반복문을 멈추세요.
# print(x)의 위치에 따라 결과값이 다르게 나온다.
for x in a:
  print(x)
  if x == 'g':
    break
```

    K
    a
    g
    


```python
for x in a:
  if x == 'g':
    break
  print(x)
```

    K
    a
    

- enumerate()


```python
alphabets = ['A','B', 'C']
for i in alphabets:
  print(i)


alphabets = ['A','B', 'C']
for index, value in enumerate(alphabets):
  print(index, value)
```

    A
    B
    C
    0 A
    1 B
    2 C
    

## 리스트 컴프리헨션
list comprehension
- 반복문을 한 줄로 푼 것. 


```python
fruits = ['apple', 'kiwi', 'mango']
newlists = []

# 알파벳 a가 있는 과일만 추출 후, 새로운 리스트에 담기
for f in fruits:
  print(f)
  if "a" in f:
    newlists.append(f)
print(newlists)
```

    apple
    kiwi
    mango
    ['apple', 'mango']
    


```python
# 리스트 컴프리헨션
newlist = [fruit for fruit in fruits] 
print(newlist)
```

    ['apple', 'kiwi', 'mango']
    


```python
newlist = [fruit for fruit in fruits if 'a' in fruit]
print(newlist)
```

    ['apple', 'mango']
    

## 0627 복습 (반복문)
- for loop and while loop


```python
for i in range(3):  # 해당 루프를 3번 돌려라
  print("Hello World")
  print("안녕하세요")
```


```python
for i in range(1000):  # 해당 루프를 3번 돌려라
  print("No : ", i+1)
  if i == 10: # 만약 i가 10이라면
    break # 멈춰
  print("Hello World")
  print("안녕하세요")
```


```python
a = "Kaggle"

for i in a: # for돌다, i in a
  print(i)
  if i == "a":
    break
```

- 리스트의 값이 존재
- 전체 총합 구하기


```python
numbers = [1, 2, 3, 4, 5]
sum = 0 # 합치려면 변수를 만들어 줘야함

for num in numbers:
  print ("numbers: ", num)  # 확인을 위해 프린트 한 것
  sum = sum + num # sum은 반복문 안에서만 돌고 있음
  print("total: ", sum) # 확인을 위해 프린트 한 것

print("---최종 결괏값---")
print(sum)
```


```python
fruits = ['apple', 'kiwi', 'mango']
newlist = []  # 반복문을 사용하기 위해 변수를 만들어주기

# apple을 꺼내서, a가 있나요? 있네요. newlist에 추가하세요.
# kiwi을 꺼내서, a가 있나요? 없네요. 그럼 넘어가세요.
# mango을 꺼내서, a가 있나요? 있네요. newlist에 추가하세요.
for fruit in fruits:  # fruits 리스트에서 요소를 꺼내서 변수 fruit에 적용한다.
  print("조건문 밖: ", fruit)
  if "a" in fruit:
    print("조건문 안으로 들어옴: ", fruit)
    newlist.append(fruit)
print(newlist)
```

### While Loop
- 조건식이 들어간다, 조건식이 참일 때만 while문이 실행된다.
- while문 : 개발 할 때 while문을 자주 사용한다. 분석할 때 거의 사용할 일이 없다. 반복문을 주로 사용.


```python
i = 1
while i < 10: # i<10 조건식, 참일 때만 while문 코드가 실행 됨, 무한루프가 실행될 수 있음, 그래서 코드로 거짓이 나와 무한루프에서 빠져나올 수 있도록 함. 
  # 코드
  print(i)
  i += 1  # 1씩 증가
  # i -= 1  # 1씩 감소
```

## 사용자 정의 함수
- 내가 필요에 의해 직접 함수를 작성
- def
-      return


```python
def add(a =0, b=1):
  c= a+b
  return c  #retun a+b 도 가능하다.
print(add(a=5, b=4))  # 내가 a값 b값 입력하기
print(add())
```


```python
def sub(a =0, b=1):
  c= a-b
  return c
print(sub())
print(sub(a=1, b=7))
```


```python
def mul(a =0, b=1):
  c= a*b
  return c
print(mul())
print(mul(a=1, b=7))
```


```python
def div(a =0, b=1):
  c= a/b
  return c
print(div())
print(div(a=1, b=7))
```


```python
def div(a, b):
  c= a/b
  return c
# print(div())
print(div(1, 7))
```

## 함수 문서화
- 키워드 : Docstring



```python
def subtract(a,b):
  """a, b를 빼는 함수

Parameters:
a (int): int형 숫자 a가 입력
b (int): int형 숫자 b가 입력

Returns:
  int: 반환값

  """
  return a-b
print(subtract.__doc__)
print(subtract(a=5, b=10))
```
