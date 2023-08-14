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

