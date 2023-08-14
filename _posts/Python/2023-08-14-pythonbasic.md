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

2. 변수와 객체

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

3. 자료형과 연산자

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

- 문자열에 가능한 연산

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