---
layout: single
title: "Python Basic"
categories: python
tags: [language, programming, python]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

# Part 02 빅데이터 분석 - 준비

## Chapter 04 파이썬 프로그래밍 기초

### 변수와 객체

- 변수
  - 값을 저장하는 메모리 공간
  - 파이썬에서는 미리 선언하지 않는다
  - 값의 자료형이 변수의 자료형을 결정한다.
- 객체
  - 속성(attribute, 내부 데이터) + 메서드(method, 내부 연산)
  - 파이썬에서는 모든 변수와 자료형이 객체로 되어 있다

```py
a = 1 + 2j # complex data, object
a.real # attribute, 1
a.conjugate() # method, a - 2j
```

### 자료형과 연산자

- 숫자형

```py
a = 123 # integer
b = 12.34 # float
c = 1 + 2j # complex
c.real # 1
c.imag # 2
c.conjugate() # 1 - 2j
abs(c) # 2.236...
d = 0o12 # octal
e = 0x12A # hexadecimal
```

- 숫자형 연산

*숫자형 연산이 속도가 빠르다*

```py
a + b # 합
a - b # 차
a * b # 곱
a / b # 나누기
a ** b # 멱
a % b # 나머지
a // b # 몫
```

- 논리형

```py
a = True
```

- 그룹 자료형

```py
a = 'Hello World' # string
b = """Hello World""" # string, several lines
```

- 문자열 연산

```py
a = 'Hello World' # string
'Hello' + ' World' # 합
'Hello' * 2 # 곱하기, 반복
a[0] # index, H
a[-1] # last index, d
a[0:2] # Slicing, 0 ~ 1
a[2:] # Slicing, 2 ~ end
a.count('l') # count
a.find('e') # find index. 없으면 -1 출력
a.index('e') # find index. 없으면 error
a.join(',') # insert H,e,l,l,o, ,W,o,r,l,d
a.upper() # capital
a.lower() # lowercase letter
a.lstrip() # remove left space
a.rstrip() # remove right space
a.strip() # remove space
a[1] = 'a' # edit string. impossible
a.replace('e', 'a') # replace string. possible
a.split() # split to list, ['Hello', 'World']
```

- 리스트

```py
a = [1, 2, 3, 4, 2, 2, 5]
b = ['life', 'is', 'too', 'short']
a[0] # indexing
a[0:2] # slicing
a + b # 리스트 연결
a[0] + b # error. a[0]는 int
[a[0]] + b # possible
a * 3 # repeat list
a[0] = 99 # edit list, possible. string의 원소 변경은 되지 않는 것과 다르게, 리스트는 변경 가능하다.
del a[-1] # delete
a.append(5) # append
a.sort() # sort
a.reverse() # reverse
a.index(3) # find index
a.insert(0, 99) # insert, index, value
a.remove(99) # remove value
a.pop() # remove last index
a.pop(0) # remove 0th index
a.count(2) # count value
```

- tuple

```py
t1 = (1, 2, 3)
t2 = (4, 5, 6)
t1[0] # indexing
t1[0:2] # slicing
t1 + t2 # 연결
t1 * 3 # repeat
t1[0] = 2 # error. 리스트와는 다르게, 튜플은 값을 변경할 수 없다
```

- dictionary

*key: value*

```py
dic = {'name': 'Hong', 'phone': '01012345678', 'birth': '0814'} # generate dictionary
dic['pet'] = 'dog' # add element
del dic['pet'] # delete element
dic['name'] # get element
dic.keys() # 
list(dic.keys()) # list of keys
dic.items() # keys and values
dic.clear() # remove all element
```

- set

```py
s1 = {1, 2, 'a', 5}
s2 = set([1, 2, 3, 4, 5, 6])
s3 = set([4, 5, 6, 7, 8, 9])
s2 & s3 # 교집합
s2.intersection(s3) # 교집합
s2 | s3 # 합집합
s2.union(s3) # 합집합
s2 - s3 # 차집합
s2.difference(s3) # 차집합
s2.add(7) # 원소 하나 추가
s2.update([6, 7, 8, 9, 10]) # 원소 여러개 추가
s2.remove(7) # 특정원소 제거
```

### 조건문과 반복문

- 조건문

```py
if condition1:
    execution1
elif condition2:
    excution2
else:
    pass # do nothing
```

- 조건문 연산

```py
==
!=
>
<
>=
<=
and
or
not
1 in [1, 2, 3] # tuple, list, string
0 not in [1, 2, 3]
'a' in ('a', 'b', 'c', 'd')
'i' not in 'Python'
```

- 반복문

```py
test_list = ['one', 'two', 'three']
for i in test_list:
    print(i + '!')
```

```py
i = 0
while i < 5:
    i += 1
    print('*' * i)
```

### 함수

- 사용자 정의 함수

```py
# 인수 개수가 정해진 경우
def sum1(a, b):
    x = a + b
    return x

# 인수 개수가 정해지지 않은 경우
def sum2(*args):
    x = 0
    for i in args:
        x += i
    return x

a = 5
b = 3
sum1(a, b)
sum2(1, 2, 3, 4, 5)
```

- 내장 함수

```py
abs(-3.5)
all([1, 2, 3, 4]) # True, 모든 원소가 True이면 True 반환
all([4, -2, 0.0, 4]) # False
any([4, -2, 0.0, 4]) # True, 하나라도 참이면 True 반환
chr(97) # 아스키코드 값 97에 대한 문자 출력, 'a'
ord('a') # 문자'a'에 대한 아스키코드 값 출력, 97
dir([1, 2, 3]) # 객체가 가진 attribute와 method 보여주기
divmod(7, 3) # 몫과 나머지를 튜플로 반환
oct(8) # 8진수로 변환
hex(16) # 16진수로 변환
# 객체의 주소 값 반환
a = 3
id(a) 
int('3') # change to int
str(3) # change to string
list('Python') # ['P', 'y', 't', 'h', 'o', 'n'], change to list
tuple('Python') # ('P', 'y', 't', 'h', 'o', 'n'), change to tuple
type('abc') # return type
# lambda
sum = lambda a, b: a + b
sum(3, 5)
max([1, 4, 2, 8, 6])
min([1, 4, 2, 8, 6])
pow(2, 4) # 제곱
c = input('insert value c') # user input
list(range(5)) # [0, 1, 2, 3, 4]
list(range(5, 10)) # [5, 6, 7, 8, 9]
list(range(5, 10, 2)) # [5, 7, 9]
len('Python') # length, 6
sorted([3, 0, 2, 1]) # ordered, [0, 1, 2, 3]
sorted('Python') # ordered and return list, ['P', 'h', 'n', 'o', 't', 'y']
```

- module and package

```py
import urllib.request # import 패키지명.모듈명
urllib.request.Request('http://www.hanb.co.kr') # 패키지명.모듈명.모듈함수
import pandas # 모듈명
pandas.DataFrame() # 모듈명.모듈함수
from datetime import datatime # from 패키지명 import 모듈명
datetime.now() # 모듈명.모듈함수
```

### 파일 처리

```py
# 파일 생성하기
f = open("D:/새파일.txt", 'w') # 파일 객체 생성
for i in range(1, 6):
    data = "%d번째 줄입니다. \n" % i
    f.write(data) # 파일 쓰기
f.close # 파일 닫기

# 파일 추가하기
f = open("D:/새파일.txt", 'a') # 파일 추가하기 모드로 열기
for i in range(6, 10):
    data = "%d번째 줄 추가입니다. \n" % i
    f.write(data) # 파일 쓰기
f.close # 파일 닫기

# 파일 읽기 1
f = open("D:/새파일.txt", 'r') # 파일 읽기 모드로 열기
line = f.readline()
print(line)
while True:
    line = f.readline()
    if not line: break
    print(line)
f.close()

# 파일 읽기 2
f = open("D:/새파일.txt", 'r') # 파일 읽기 모드로 열기
lines = f.readlines()
print(lines)
for line in lines:
    print(line)
f.close()

# 파일 읽기 3
f = open("D:/새파일.txt", 'r') # 파일 읽기 모드로 열기
data = f.read()
data
f.close()

# 파일 처리 후 파일 닫기(자동)
with open("D:/새파일.txt", 'w') as f:
    f.write("Now is better than never.")
```

### 데이터 분석을 위한 주요 라이브러리

```bash
pip install numpy
```

```py
import numpy as np
ar1 = np.array([1, 2, 3, 4, 5])
ar2 = np.array([1, 2, 3], [10, 20, 30])
ar3 = np.arange(1, 11, 2) # array([1, 3, 5, 7, 9])
ar4 = np.array([1, 2, 3, 4, 5, 6]).reshape(3, 2) # [1,2], [3, 4], [5, 6]
np.zeros((2,3))
ar2[0:2, 0:2] # slicing
ar1 + 10
ar1 - 5
ar1 * 3
ar1 / 2
np.dot(ar2, ar4) # inner product
```

```bash
pip install pandas
# series, dataframe
```

```py
import pandas as pd
# Series
data1 = [10, 20, 30, 40, 50]
data2 = ['1반', '2반', '3반', '4반', '5반']
sr1 = pd.Series(data1)
sr2 = pd.Series(data2)
sr3 = pd.Series([101, 102, 103, 104, 105])
sr4 = pd.Series(['월', '화', '수', '목', '금'])
sr5 = pd.Series(data1, index = [1000, 1001, 1002, 1003, 1004]) # index change
sr6 = pd.Series(data1, index = data2)
sr7 = pd.Series(data2, index = data1)
sr8 = pd.Series(data2, index = sr4)
sr8[0:4] # slicing
sr8.index
sr8.values
sr1 + sr3 # 둘 다 숫자이므로 덧셈 연산 수행
sr4 + sr2 # 둘 다 string이므로 문자연결 수행

# DataFrame
# Dictionary to dataframe
data_dic = {'year': [2018, 2019, 2020], 'sales': [350, 480, 1099]}
df1 = pd.DataFrame(data_dic)
# List to dataframe
df2 = pd.DataFrame([[89.2, 92.5, 90.8],[92.8, 89.9, 95.2]], index = ['중간고사', '기말고사'], columns = data2[0:3])
df2

data_df = [['20201101', 'Hong', '90', '95'],['20201102', 'Kim', '93', '94'],['20201103', 'Lee', '87', '97']]
df3 = pd.DataFrame(data_df)
df3.columns = ['학번', '이름', '중간고사', '기말고사']
df3
df3.head(2)
df3.tail(2)
df3('이름') # column '이름'
df3.to_csv('path and filename', header='False')
df4 = pd.read_csv('path and filename', encoding='utf-8', index_col = 0, engine='python')
```

```bash
pip install matplotlib 
```

```py
import matplotlib.pyplot as plt

# line plot
x = [2016, 2017, 2018, 2019, 2020]
y = [350, 410, 520, 695, 543]
plt.plot(x, y)
plt.title('Annual sales')
plt.xlabel('years')
plt.ylabel('sales')
plt.show()

# bar plot
y1 = [350, 410, 520, 695]
y2 = [200, 250, 285, 350]
x = range(len(y1))
plt.bar(x, y1, width = 0.7, color = "blue")
plt.bar(x, y2, width = 0.7, color = "red")
plt.title('Quarterly sales')
plt.xlabel('Quarters')
plt.ylabel('sales')
xLabel = ['first', 'second', 'third', 'fourth']
plt.xticks(x, xLabel, fontsize = 10)
plt.legend(['chairs', 'desks'])
plt.show()
```