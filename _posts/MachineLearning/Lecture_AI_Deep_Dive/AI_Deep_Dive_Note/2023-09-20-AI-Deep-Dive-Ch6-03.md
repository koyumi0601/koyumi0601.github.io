---
layout: single
title: "AI Deep Dive, Chapter 6. 인공신경망 그 한계는 어디까지인가? 03. Beautiful Insights for ANN (AI가 스스로 학습을 한다는 것의 실체는)"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 6 - 03. Beautiful Insights for ANN (AI가 스스로 학습을 한다는 것의 실체는)

- ANN: Artificial Neural Network

- MLP는 행렬 곱하고, 벡터 더하고, activation, 행렬 곱하고 벡터 더하고 activation, ...

- 인공 신경망은 함수다

- 어떤 연속 함수든 다 표현 가능하다. Universal approximation theorem

  - 표현에 한계가 있어서 못 찾는 경우는 없다는 것
  - 효율만 고려해서 짜면 된다

- 원하는 출력 나오도록 (L을 줄이도록)

  조금씩 가중치 값들을 update (gradient 반대 방향으로) 해 나가는 것.

  그게 AI가 학습을 한다는 것의 실체다
