---
layout: single
title: "AI Deep Dive, Chapter 5. 이진 분류와 다중 분류 06. Summary (인공신경망에 대한 정리)"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 5 - 06. Summary (인공신경망에 대한 정리)

- Step 1. 입, 출력 정의
  - 문제 정의: 내가 어떤 문제를 풀 것인가? 
    - 회귀(키/몸무게) / 이진 분류(강아지/고양이) / 다중 분류(강아지/고양이/소) / multi-label 분류(영화 액션,로맨스,드라마 [1, 1, 0]) / ...
    - 정답의 분포를 무엇으로 가정할 것이며 출력은 무엇으로 볼 것인가?
      - 베르누이: -log q, 가우시안: MSE, Multinomial 혹은 categorial distribution: cross entropy
      - 결국 MLE
  - 출력
    - 가우시안: 평균 (분산도 학습시킬 수 있음)
    - 다중분류: 분포값
- Step 2. 모델 만들기
  - 이런 문제를 풀기에 어떤 모델이 적합할까?
    - MLP / CNN / RNN / ...
- Step 3. Loss 정의
  - 회귀면 MSE, 분류면 cross-entropy (사실은 다 NLL이지만)
- Step 4. weight 최적화
  - gradient descent 기반의 최적화 알고리즘들 (SGD, mini-batch SGD, moment, RMSProp, ADAM)
