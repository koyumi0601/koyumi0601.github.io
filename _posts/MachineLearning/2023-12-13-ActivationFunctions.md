---
layout: single
title: "[Study Summary] Activation Functions"
categories: machinelearning
tags: [ML, Machine Learning, AI, Summary]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true



---

*Activation Functions for machine learning*

강의 노트가 아니라 정보만 정리한 것

![activation_fun]({{site.url}}\images\2023-12-13-ActivationFunctions\activation_fun.png)





1. **이미지 분류 (Image Classification)**:
   - 주로 사용되는 활성화 함수: ReLU, Leaky ReLU, Swish
   - 설명: 이미지 분류 작업에서는 주로 ReLU, Leaky ReLU, Swish와 같은 활성화 함수가 사용됩니다. 이들은 이미지 분류 문제에서 잘 동작하며, 합성곱 신경망 (CNN)과 함께 주로 사용됩니다.
2. **이미지 분할 (Image Segmentation)**:
   - 주로 사용되는 활성화 함수: ReLU, Leaky ReLU, Swish
   - 설명: 이미지 분할은 객체의 경계를 감지하는 작업으로, 이미지 분류와 비슷한 활성화 함수를 사용합니다. 주로 CNN과 함께 사용됩니다.
3. **객체 검출 (Object Detection)**:
   - 주로 사용되는 활성화 함수: ReLU, Leaky ReLU, Swish
   - 설명: 객체 검출은 이미지에서 객체의 위치와 경계 상자를 예측하는 작업입니다. 이미지 분류와 유사한 활성화 함수가 주로 사용됩니다.
4. **이미지 캡셔닝 (Image Captioning)**:
   - 주로 사용되는 활성화 함수: LSTM, GRU, Swish
   - 설명: 이미지 캡셔닝은 이미지에 대한 설명을 생성하는 작업으로, 순환 신경망 (RNN)과 함께 LSTM, GRU와 같은 활성화 함수가 사용됩니다.
5. **비주얼 임베딩 (Visual Embedding)**:
   - 주로 사용되는 활성화 함수: ReLU, Swish
   - 설명: 비주얼 임베딩은 이미지를 고정 차원 벡터로 변환하는 작업으로, ReLU 및 Swish와 같은 활성화 함수를 사용한 신경망이 주로 활용됩니다.
6. **생성 모델 (Generative Models)**:
   - 주로 사용되는 활성화 함수: ReLU, Leaky ReLU, Swish (생성자), Sigmoid (판별자)
   - 설명: 생성 모델은 이미지 생성 작업에 사용되며, 생성자 네트워크에서는 ReLU, Leaky ReLU, Swish와 같은 활성화 함수가, 판별자 네트워크에서는 Sigmoid가 주로 사용됩니다.



