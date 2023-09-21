---
layout: single
title: "AI Deep Dive, Chapter 2. 왜 현재 AI가 가장 핫할까? 03. 머신러닝의 분류, 지도학습과 비지도학습"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 2 - 03. 머신러닝의 분류, 지도 학습과 비지도 학습



##### 머신러닝의 네 가지 분류

![category]({{site.url}}/images/$(filename)/category.png)





지도 학습 중에서도 딥러닝인 지도학습도 있고



지도 = 가르친다. 정답(label)을 알고 있다. 강아지 1 고양이 0 레이블링.



예시

 - 회귀 regression

 - 분류 classification

   ![classification]({{site.url}}/images/$(filename)/classification.png)

   - classification

   - localization

   - object detection (자율 주행에서 많이 쓰임)

   - Instance Segmentation (pixel by pixel로 분류하는 것, 정밀하게 영역을 나눠줄 수 있다. 강아지 1, 강아지 2 구분도 가능 함.)

   - 포즈도 알려줄 수 있다. 머리가 어디, 목이 어디.

     ![classification_pose]({{site.url}}/images/$(filename)/classification_pose.png)



정답을 알고 있는 데이터가 굉장히 많아야 한다.



표정 - 랜드마크

![landmark]({{site.url}}/images/$(filename)/landmark.png)





비지도 학습



- 반대로, 정답을 모른다

- 군집화 (K-means, DBSCAN, ...)

  - 키-몸무게 그룹핑

    ![height_weight]({{site.url}}/images/$(filename)/height_weight.png)

- 차원 축소 (데이터 전처리: PCA, SVD, ...)

  - 데이터가 퍼져있을 때, 축을 하나 그어주고, 그 축 값을 읽어준다. 2D 데이터가 1D 데이터가 됐다. 데이터 용량이 워낙에 크기 때문에 전처리에 활용해야 한다.

    ![dimension_shirink]({{site.url}}/images/$(filename)/dimension_shirink.png)

- GAN (딱 비지도 학습이라 하기 약간 애매)
