---
layout: single
title: "AI Deep Dive, Chapter 7. 깊은 인공신경망의 고질적 문제와 해결 방안 01. 직관적으로 이해하는 vanishing gradient (식당, 대기업, 스타트업)"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 7 - 01. 직관적으로 이해하는 vanishing gradient (식당, 대기업, 스타트업)

- 무턱대고 깊게 만들면, 여러가지 문제가 생긴다

  - vanishing gradient

  - overfitting

  - loss landscape가 꼬불해지는 문제, loss함수의 개형이 꼬불꼬불해진다는 거

- Vanishing gradient

  - layer가 많으면 입력 층에 가까울수록 미분이 사라진다

  - $$ w_{k+1} = w_k - \alpha G $$ G가 0에 가까워서, 업데이트가 안되는 것. 입력층 근처에서

  - 왜?
    - 앞으로 갈 수록, 미분값이, 액웨액웨액앤...인데, 액티베이션 함수의 미분값이 너무 작음. 
      - ex. sigmoid 사용 시, 최대 기울기는 1/4 운 좋으면...
        - 그럼 1/4보다 작은 값들이 계~~~~속 곱해지니까 0에 점점 가까워짐. 액이 많이 들어갈 수록. 0에 가까워진다

  - 주범은 sigmoid
  - gradient 소멸 -> update x -> 앞단에 이상한 weight -> 깊은 게 오히려 독이 됨
  - 직관: 앞에서 입력 데이터를 망쳐 놓으면 뒤에서 손 쓸 방법이 없다.
    - 식당으로 치면, 재료 -> 재료 손질팀 -> 요리 -> 플레이팅
      - 재료 손질팀은 애기 - 학습이 이루어지지 않음
      - 요리 팀은 업데이트가 그래도 좀 됨, 초딩
      - 플레이팅 팀은 업데이트가 많이 됨. 고딩
    - 회사로 비유: 대기업 vs 스타트업 생각 (입력 쪽이 사원, 출력 쪽이 임원급)
      - 반도체 재료 - 사원 - 부장 - 임원
        - 대기업은 계급이 너무 많음 vs 스타트업은 layer가 적어서 이런 문제가 x
    - 오히려 underfitting이 일어난다고 표현 함. 
      - training data에서도 못한다

  
