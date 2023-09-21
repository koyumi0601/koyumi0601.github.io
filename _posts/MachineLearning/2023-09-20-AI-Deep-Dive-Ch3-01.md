---
layout: single
title: "AI Deep Dive, Chapter 3. 왜 우리는 인공신경망을 공부해야 하는가? 01. 인공신경망, weight와 bias의 직관적 이해, 인공신경망은 함수다"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 3 - 01. 인공신경망, weight와 bias의 직관적 이해, 인공신경망은 함수다



신경

![neuron]({{site.url}}/images/$(filename)/neuron.png)



단순화한 모델



![neuron1]({{site.url}}/images/$(filename)/neuron1.png)

동그라미 - 노드

작대기 - 에지



뒷통수 0, 손바닥 0 - 누워있어서 자극이 없다

뒷통수 0, 손바닥 1 - 하이파이브

뒷통수 1, 손바닥 0 - 뒷통수 맞음

뒷통수 1, 손바닥 1 - 구타



![neuron2]({{site.url}}/images/$(filename)/neuron2.png)

들어오는 자극의 세기를 보고, 전달할 지 말지를 결정한다. -> 활성화 함수라고 한다.

음수 값 -> 전달하지 않는다

양수 값 -> 전달한다.

출력은 0또는 1이다.



이 출력이, 위기상황을 학습한다고 쳤을 때, 

출력 0: 위기 x

출력 1: 위기 o



자고 있을 때, 뒷통수 0, 손바닥 0 - 활성화 함수 - 출력 0

모기, 뒷통수 0, 손바닥 0.000001 - 활성화 함수 1 - 출력 1 (너무 민감...) -> 둔감하게 만든다. bias이용. 역치 -2만큼 둔감화.



![neuron3]({{site.url}}/images/$(filename)/neuron3.png)



그러면 모기의 경우, activation 함수가 활성화되지 않는다. 위기가 아니다.



아직 부족..



뒷통수 1, 손바닥 0 -> 위기가 아니다고 해버림



weight = 중요도



![neuron4_weight]({{site.url}}/images/$(filename)/neuron4_weight.png)



weight를 0을 주면 마비일 것





뒷통수 1 * 뒷통수 가중치 10 + 손바닥 0 * 손바닥 가중치 1 - 2 (역치, bias) = 8 -> 활성화 -> 위기이다.





여러 가지 activation이 존재. 여기서 본 건 unit step function



주어진 입력(강아지)에 대해서 원하는 출력(1)이 나오도록 웨이트, 바이어스를 정해줘야 한다.
