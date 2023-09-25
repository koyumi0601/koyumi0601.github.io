---
layout: single
title: "AI Deep Dive, Chapter 3. 왜 우리는 인공신경망을 공부해야 하는가? 02. 선형 회귀, 개념부터 알고리즘까지 step by step"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*




# Chapter 3 - 02. 선형 회귀, 개념부터 알고리즘까지 step by step



### 입력과 출력 간의 관계를 선형으로 놓고 추정하는 것

-> 처음보는 입력에 대해서도 적절한 출력을 얻기 위함



- 키와 몸무게의 관계를 ax+b로 놓고 a, b를 잘 추정해서

  -> 처음보는 키에 대해서도 적절한 몸무게를 출력하는 머신을 만들어보자

​		-> 즉, 알아내야 할 것은 최적의 웨이트, 바이어스

​			-> 뭐에 기반해서 알아낼까? 데이터 기반

​				-> 선형회귀는 지도학습에 해당 함

​			-> 최적의 a,b? 내가 고른 a, b가 좋다 나쁘다를 판단할 수 있어야 함
