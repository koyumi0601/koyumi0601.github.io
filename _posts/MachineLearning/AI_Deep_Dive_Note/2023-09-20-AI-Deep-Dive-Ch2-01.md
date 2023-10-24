---
layout: single
title: "AI Deep Dive, Chapter 2. 왜 현재 AI가 가장 핫할까? 01. AI vs ML vs DL (Rule-based vs Data-based)"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*AI Deep Dive Note*



# Chapter 2 - 01. AI vs ML vs DL (Rule-based vs Data-based)



- AI 인간의 지능을 인공적으로 만든 것
- How? 인간은 어떻게 알까?
- 인간의 사고방식을 흉내내자! 이것이 딥러닝의 핵심



AI 범위 > ML > DL

- AI

  - 규칙 기반 알고리즘
    - 강아지 vs 고양이
      - 강아지는 코가 검고, 고양이는 핑크고..
    - 알고리즘 만드는 사람이 공부를 해야 했다. 코가 어떻고..

  - 결정 트리, 선형 회귀, 퍼셉트론, SVM (Support Vector Machine)

  - CNN, RNN, GAN

- ML: Data 기반
  - 강아지 vs 고양이
    - 여러 강아지 사진을 보여줌. 강아지라고 알려줌. 사진을 뒤집어서도 보여줌. 검은 배경에 검은 강아지 사진도 보여줌. 노을에 비친 강아지도 보여줌(털 색깔 변형 옴). 두 발로 서있는 강아지도 보여줌. 누운 강아지도 보여줌. 천에 가려진 강아지도 보여줌
    - 고양이도 숨어있는 고양이, 노을에 그림자만 보이는 고양이, 누운 고양이, 오드아이 고양이 다 보여줌.
  - 머신이 공부를 했다.
  - 데이터가 많을 수록 정확도가 높아짐.
  - 학습 데이터 외의 테스트 데이터를 보여줬을 때 맞출 수 있어야 한다.

- DL: Deep Neural Network로 학습을 하면 DL
  - 입력, 인공신경망, 깊은 인공신경망을 활용

![DNN]({{site.url}}/images/$(filename)/DNN.png)