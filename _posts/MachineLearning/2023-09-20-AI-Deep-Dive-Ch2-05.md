---
layout: single
title: "AI Deep Dive, Chapter 2. 왜 현재 AI가 가장 핫할까? 05. 강화학습"
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



# Chapter 2 - 05. 강화학습

강아지 "손" 훈련

손을 얹으면 간식을 준다

손을 얹지 않으면 간식을 주지 않는다



Agent: 강아지

Reward: 간식

Environment: 보호자 (간식 주는 사람, 특정 상황에 따라서 reward를 주는 것)

Action: 손 올리기



Agent는 reward를 최대한 얻고 싶어한다는 것이 기본 전제이다.



알파고



오목

- Agent: 흑돌

- Reward: 승리

- Environment: 흰돌

- Action: 돌을 두는 것, 수

- 한개일 땐 두개, 두 개일땐 세개를, ..., 다섯개를 완성시켜야 한다는 것을 학습



미로찾기

![maze]({{site.url}}/images/$(filename)/maze.png)

goal로 가면, R=100, 지옥 R=-100 이건 설정을 여러분이 잘 해줘야 함



Agent: 

Reward:

Environment:

Action: 수

State: 바둑판 상황, 위치

Q-function: Q(Current State, Current Action) - 현재 상황에서 현재 액션은 몇 점인지 평가

Episode: 게임에서 한판 지고, 이기고 하는 것과 같은 개념. 시도. 대신 episode마다 q값은 저장해놓을 수 있음.



전후좌우로 움직이는 것에 대해 reward를 설정한다.

![maze_q]({{site.url}}/images/$(filename)/maze_q.png)



첫 번째 에피소드에서, 오른쪽으로 움직여서 지옥으로 가면 R-100, 오른쪽으로 가는 액션에 -100의 q값을 저장한다.

Reward를 전파하여 q값을 넣는다. 근처 칸의 4개 중 가장 큰 값을 전달



![maze_q_reward]({{site.url}}/images/$(filename)/maze_q_reward.png)



그러면, 에피소드 100, 100이 되면 q값이 다 있으니까 무조건 정답 길로 간다.



지금 설명은 좀 부족.. 길이 좀 더 있다고 해보자.

처음에는 우연히 가게되는데, 정답 길에 100을 쓰면, 에피소드 백번 천번해도 돌아들어간다. 최적이 아니다.

![maze_not_eff]({{site.url}}/images/$(filename)/maze_not_eff.png)



좋은 새로운 path를 알아내려면, Exploration 탐험한다고 함. 입실론 그리디 epsilon greedy

입실론 = 0.1 

Q를 믿지 말고, 확률 0.1만큼은 다른데로 가, 0.9만큼은 greedy하게 Q대로 가

우연히 아래를 고른다면, 새로운 길을 찾게 된다.

![epsilon_greedy]({{site.url}}/images/$(filename)/epsilon_greedy.png)



Discount factor : 감마, 0-1 사이의 값. 예 0.9



어느 길이 더 좋은 지 모름

Reward를 가지고 올 때마다 discount factor를 곱해서 가져온다. 

![discount_factor]({{site.url}}/images/$(filename)/discount_factor.png)

![discount_factor2]({{site.url}}/images/$(filename)/discount_factor2.png)

갈림길에서 아래로 간다.
