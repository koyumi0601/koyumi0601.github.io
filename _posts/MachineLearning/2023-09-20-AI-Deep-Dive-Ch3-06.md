---
layout: single
title: "AI Deep Dive, Chapter 3. 왜 우리는 인공신경망을 공부해야 하는가? 06. Mini-batch SGD"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 3 - 06. Mini-batch SGD



- SGD: 하나만 보는 건 너무 성급한 거 아니냐? -> mini batch! epoch별로 2개씩 보자

![ch0306_1]({{site.url}}/images/$(filename)/ch0306_1.png)



- GPU는 병렬연산이 가능하여 여러 데이터에 대해서도 빠르다

![ch0306_2]({{site.url}}/images/$(filename)/ch0306_2.png)

- 미니 배치 사이즈를 키울 수록 좋을까? 8k까지만!

![ch0306_3]({{site.url}}/images/$(filename)/ch0306_3.png)

[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://aiichironakano.github.io/cs653/Goyal-LargeMinibatchSGD-arXiv17.pdf)

- 제안 1. batch size를 두 배 키웠으면, learning rate도 두 배 키워라

- 제안 2. warm-up



# 미니배치 설명

미니배치 크기(Mini-batch Size)는 딥 러닝에서 학습 데이터를 나눠서 처리하는 방법 중 하나로, 전체 데이터셋을 한 번에 처리하는 것이 아니라 작은 미니배치 단위로 데이터를 나누어 학습합니다. 미니배치 크기는 학습 알고리즘의 하이퍼파라미터 중 하나로 설정되며, 이 크기를 조절하여 학습 과정의 속도와 안정성을 조절할 수 있습니다.

미니배치 크기의 장점은 다음과 같습니다:

1. 학습 속도 향상: 전체 데이터셋을 한 번에 처리하는 것보다 미니배치 단위로 나눠서 처리하면 학습 과정이 빨라집니다. 특히 대규모 데이터셋에서 이점을 얻을 수 있습니다.
2. 메모리 효율성: 전체 데이터를 메모리에 한 번에 로드할 필요가 없으므로 메모리 효율적으로 학습할 수 있습니다.
3. 더 나은 일반화: 미니배치 학습은 데이터를 무작위로 섞는 효과를 가지므로 모델이 전체 데이터에 과적합되는 것을 방지하고 일반화 성능을 향상시킬 수 있습니다.

GPU 병렬 연산은 그래픽 처리 장치(GPU)를 사용하여 딥 러닝 모델의 학습 및 추론 과정을 가속화하는 기술입니다. GPU는 병렬 처리 능력이 뛰어나기 때문에 많은 수치 연산을 동시에 처리할 수 있어 딥 러닝 모델의 학습 속도를 향상시킵니다.

미니배치 크기와 GPU 병렬 연산은 학습 속도와 메모리 사용량을 관리하고 모델 성능을 향상시키는 데 중요한 역할을 합니다. 대규모 데이터셋과 복잡한 모델의 경우 적절한 미니배치 크기와 GPU 가용성을 고려하여 학습을 조정하면 학습 과정이 효율적으로 진행됩니다.