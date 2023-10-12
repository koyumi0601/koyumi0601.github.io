---
layout: single
title: "AI Deep Dive, Chapter 5. 이진 분류와 다중 분류 04. 인공신경망은 MLE 기계다"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 5 - 04. 인공신경망은 MLE 기계다

- maximum likelihood estimation

- $$ (q-1)^2 $$ vs $$ -log q $$를 최소화하는 것의 뿌리는 모두 MLE이다.

- $$ q^y(1-q)^{1-y} $$를 다시 보자.. -> 베르누이 분포 식. 이것을 likelihood로 삼자.



## 베르누이 분포식

- ![ch0504_1]({{site.url}}/images/$(filename)/ch0504_1.png)

![ch0504_2]({{site.url}}/images/$(filename)/ch0504_2.png)



- 이 식을 likelihood로 삼자
