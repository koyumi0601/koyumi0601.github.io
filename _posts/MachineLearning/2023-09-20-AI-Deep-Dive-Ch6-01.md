---
layout: single
title: "AI Deep Dive, Chapter 6. 인공신경망 그 한계는 어디까지인가? 01. Universal Approximization Theorem (왜 하필 인공신경망인가)"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 6 - 01. Universal Approximization Theorem (왜 하필 인공신경망인가)

- Universal Approximization Theorem
  - 한계가 없다는 것을 이론적으로 보여줌
  - MLP: 행렬 곱하고 벡터 더하고 activation, ... 이런 함수다
  - $$ f(f(xW_1 + b_1)W_2+b_2)) $$. (1차 함수에 대하여 activation)
  - 굳이 이런 형태로 함수를 표현해야 하는 이유? x^Tx = x^2 이런 건 왜 안쓰나? 히든 레이어 딱 한층만 있어도 어떤 연속 함수든 나타낼 수 있기 때문. 즉, loss를 딱 0으로 만들어 버릴 수 있다.
    - 

