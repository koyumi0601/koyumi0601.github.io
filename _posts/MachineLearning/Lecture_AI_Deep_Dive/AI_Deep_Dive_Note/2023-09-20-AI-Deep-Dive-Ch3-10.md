---
layout: single
title: "AI Deep Dive, Chapter 3. 왜 우리는 인공신경망을 공부해야 하는가? 10. K-fold Cross Validation"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 3 - 10. K-fold Cross Validation

- training, validation, test data로 나눌만큼 데이터 수가 많지 않을 때
- 예를 들어 전체 데이터가 120개 일 때, - 뭘 해도 잘 안됨..



![ch0309_0]({{site.url}}/images/$(filename)/ch0309_0.png)





##### 필기

![ch0310_1]({{site.url}}/images/$(filename)/ch0310_1.png)



- 순차로 선택하진 말자. ex. 1-20.jpg (강아지) -> 편향
- 데이터 셋을 5개의 조합으로 만든 후, validation loss의 평균을 내자 

![ch0310_2]({{site.url}}/images/$(filename)/ch0310_2.png)

- 가장 validation loss 평균이 작은 hyperparameter set을 고르는 데 사용 가능

  





![ch0310_3]({{site.url}}/images/$(filename)/ch0310_3.png)



- k-fold 활용

  - 산출된 hyperparameter set을 이용해 전체 모델을 다시 한번 training 시킨다 

  - 혹은 모델이 5개다 라고 가정하고, 다수결로 결정할 수 있다.

    

![ch0310_4]({{site.url}}/images/$(filename)/ch0310_4.png)
