---
layout: single
title: "AI Deep Dive, Chapter 5. 이진 분류와 다중 분류 03. MSE vs likelihood(왜 log-likelihood를 써야할까?)"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 5 - 03. MSE vs likelihood(왜 log-likelihood를 써야할까?)



- 의문) 앞서 배운 MSE를 여기에도 그대로 (loss함수로) 적용하면 안될까?
  - $$ (q-1)^2 $$ 을 minimize
  - $$ -logq $$를 minimize
- 되긴 되는데 학습 성능이 떨어진다.
  - $$ -log q $$의 경우, 강아지(1)인데 예측값이 고양이(0)이 나왔다면, 에러(loss)함수가 무한대가 된다.
    - **민감도**가 높다
  - MSE를 사용하면 에러가 1밖에 안됨

![ch0503_1]({{site.url}}/images/$(filename)/ch0503_1.png)



- 마지막 sigmoid 통과 직전 weight에 대해 loss function의 개형이 $$ (q-1)^2 $$ 라면, non-convex, $$ -log q $$라면 convex이다.
  - convex: 두 번 미분해서 양수면 아래로 볼록, ex. $$ x^2 $$

![ch0503_2]({{site.url}}/images/$(filename)/ch0503_2.png)
