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



### 신경

![neuron]({{site.url}}/images/$(filename)/neuron.png)



### 단순화한 모델



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



근데, AI가 스스로 적절한 웨이트, 바이어스를 알아낸다. 어떻게?





자극을 전달 받는 부위로부터, 신경 다발을 거쳐서, 뇌까지 전달 된다.

멀리 있는 신경에 대해서는 (발바닥) -> 허리 -> 머리



중간 전달하는 척수들의 층이 있다. 거기도  활성화 함수가 들어간다.



연습삼아, weight와 bias 개수를 세어보자



3 - 2 - 1

3x2+2   +   2x1 + 1

노드 개수 * 웨이트 개수, activation 에 bias하나씩 붙음

![wb]({{site.url}}\images\2023-09-20-AI-Deep-Dive-Ch3-01\wb.png)



input layer & output layer & hidden layer

- input layer: 입력 들어가는 층,  숫자만 있는 거
- output layer: weight랑 합쳐서, bias랑 마지막 아웃풋 노드까지
- hidden layer: 중간 층. 층의 마지막은 노드까지임.



깊은 인공 신경망은? DNN Deep Neural Network

노드끼리 싹다 연결되어 있으면 fully connected layer라고 한다



모든 layer가 fully connected layer로 구성되어 있으면 MLP(multi-layer perceptron) 이라고 부름



원래 perceptron은 unit step function을 활성화 함수로 사용하는 신경망 하나를 의미한다.



그렇지만 MLP는 임의의 활성화 함수를 사용하는 인공신경망을 말하는 것으로 의미가 확장되었다.





인공 신경망은 함수다. 

다시말해, 인공 신경망은 입력과 출력을 연결시켜주는 연결 고리인 것이고

결국, 주어진 입력에 대해서 원하는 출력이 나오도록하는 함수를 찾고 싶은 것



그럼, ax+b 일차함수를  인공신경망으로 표현해보자 linear activation



