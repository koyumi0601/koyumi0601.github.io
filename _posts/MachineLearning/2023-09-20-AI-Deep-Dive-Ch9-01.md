---
layout: single
title: "AI Deep Dive, Chapter 9. 왜 RNN보다 트랜스포머가 더 좋다는 걸까?"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



## 01 연속적인 데이터예시와 행렬과 벡터 식으로 RNN 이해하기

- 문장을 넣어서, 긍정 vs 부정 이진분류
- 주가를 넣어서, 내일 주가를 알려줘
- 동영상을 넣어서 (프레임의 연속) 어떤 동작이냐 분류 혹은 동작 점수를 알려줘
- 저는 강사 입니다
  - 단어 -> 숫자로 바꾼다 (word embedding)
    - 저는 [1, 0, 0] x_1
    - 강사 [0, 1, 0] x_2
    - 입니다 [0, 0, 1] x_3
  - Neural Network에 하나씩 순차적으로 넣어보자 (RNN의 접근법)

- RNN 동작 방식
  - RNN: Recurrent Neural network

![ch9_01]({{site.url}}\images\2023-09-20-AI-Deep-Dive-Ch9-01\ch9_01.png)

- 이렇게 함으로써 얻는 효과는?
  - h2의 출력을 만들 때, h1도 고려한다.
  - h3의 출력을 만들 때, h1, h2도 고려한다. 

- 수식

![ch9_02]({{site.url}}\images\2023-09-20-AI-Deep-Dive-Ch9-01\ch9_02.png)



> 추가 조사
>
> - 순환 신경망(RNN)은 **순차 데이터나 시계열 데이터를 이용하는 인공 신경망 유형**입니다. 이 딥러닝 알고리즘은 언어 변환, 자연어 처리(nlp), 음성 인식, 이미지 캡션과 같은 순서 문제나 시간 문제에 흔히 사용됩니다.
>
> - **입력과 출력을 시퀀스 단위로 처리** (시퀀스: 문장 같은 단어가 나열된 것)
>
> - 메모리 셀: RNN의 은닉층에서 활성화 함수를 통해 결과를 내보내는 역할을 하는 노드를 셀이라고 합니다. 이 셀은 이전의 값을 기억하려고 하는 일종의 메모리 역할을 수해하므로 **메모리 셀**이라고 부릅니다.
> - **은닉층의 메모리 셀에서 나온 값이 다음 은닉층의 메모리 셀에 입력됩니다.** **이 값을 은닉 상태라고 합니다.**
> - ![img](https://blog.kakaocdn.net/dn/pSQYH/btrdf2YygE3/xrhyjKgsKwrf6VQOPcn2b1/img.png)







## 02 RNN의 backpropagation과 구조적 한계를 직관적으로 이해하기



## 03 Plain RNN 실습



## 04 seq2seq 개념 및 문제점과 실습까지



## 05 RNN attention의 문제점과 트랜스포머의 self-attention



## 06 강의 마무리 (딥러닝 연구는 뭘 잘해야 할까)