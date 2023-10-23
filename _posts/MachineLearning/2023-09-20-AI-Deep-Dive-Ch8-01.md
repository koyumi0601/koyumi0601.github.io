---
layout: single
title: "AI Deep Dive, Chapter 8. 왜 CNN이 이미지 데이터에 많이 쓰일까? "
categories: machinelearning
tags: [ML, Machine Learning, AI, AI Deep Dive]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*AI Deep Dive Note*



## 01. CNN은 어떻게 인간의 사고방식을 흉내냈을까?

- 이미지 데이터에 대해서 잘 동작 함
- ex. object detection
- 신경다발을 잘 끊어냄 (FC-MLP 대비 잘 끊어냄) - 이미지를 인식할 때 뇌의 일부분만 활성화 되더라
- 첫 layer에서 모든 값을 다 보는 게 좋을까? 아니다
- 위치별 특징을 추출함
  - 위치 정보를 유지한 채로 특정 패턴(특징)을 찾는다
  - 패턴? 컨볼루션은 위치별 패턴을 찾는 연산이므로, 신경망에 적용함으로써 위치별 패턴이 있다는 사전 정보를 잘 준 것이다.
  - FC layer는 픽셀 위치를 서로 막 바꿔 학습하는 것과 다를 바가 없음
- convolution으로 얻는 효과는?
  - 가까이 있는 것들만 연결 = 위치 정보를 잃지 않음 -> 담당 노드라는 의미가 생김
    - 퍼즐 처럼 섞으면 결과가 달라진다. FC랑 다르게
- 2x2 커널에 bias도 포함해서 총 5개 값을 학습한다


## 02. CNN은 어떻게 특징을 추출할까?

- 세로 패턴이 있는 이미지
  - 내적을 생각하면 알 듯이, 필터와 비슷한 패턴이 나왔을 때 값이 크다. 
- 가로 패턴이 있는 이미지
- 실제 이미지에 적용해보니, 엣지다
- 이미지에 변형을 가하기도 한다. Blur -> 커널에 따라서, box car 5x5

- 여러 필터링 결과물을 결합할 때에는, 옆으로 쌓는 게 아니라, depth 방향으로 쌓는다. 그래야 위치 정보가 남아있다. 이걸 feature map이라고 부른다
  
### convolution layer가 FC layer와 다른 점
- 주변만 연결 (신경망을 잘 끊어서, 위치 정보를 잃지 않는다)
- weight 재사용 (쭉 긁으면서 스캔하니까 눈이 바닥에 있어도 인식)
- 여러 종류의 필터로 여러 종류의 특징을 추출
- 각 필터가 어떤 특징을 추출하는 지를 머신이 학습한다.
  - 가로필터, 세로필터, 대각선필터... 이걸 지정하는 게 아니라 학습하는 것
- Q) conv layer로 FC layer 표현할 수 있을까?
  - 100x100 입력 이미지에, kernel size 100x100, filter 종류 10개 사용하는 conv layer는 10000->10인 FC layer와 동일

## 03. 3D 입력에 대한 convolution
- 컬러사진이면 RGB 3개 채널
  - 무조건 앞에 거 채널 수에 맞춘다 = 필터도 3개 채널임. 3x5x5 -> 75개의 값들이 더해져서 하나의 값으로 나온다 / 혹은 depth wise convolution, RGB 따로 처리하는 방법도 많이 쓴다.
  - 필터 여러 개에 의해서 여러 개의 피쳐맵이 나온다







# 04 Padding, Stride and Pooling



- Padding
  - edge를 0으로 채움. 
  - 처리된 이미지의 사이즈를 유지하기 위함
  - zero padding / zero insertion
    - zero padding: 주변 값을 0으로 채움. 원치 않는 에지가 생김
    - zero insertion: 데이터 샘플링 비율을 높이기 위해 (업샘플) 중간 사이사이에 0을 채워 넣음. 후에 low pass filtering을 통해서 적절히 0 값을 보간해 줌.
- Stride
  - pooling에서 건너 뛰는 값
- Pooling
  - 사이즈를 줄여 넓은 범위를 대표할 수 있게 함. 파라미터 필요 없음
  - average pooling - 대표값으로 평균값
  - max pooling - 대표값으로 최대값
  - size=(4, 4)=이미지사이즈면, global average pooling(GAP)라고 부름
  - depth 방향(채널수)는 유지이고, row, column만 리사이즈 함



# 05 CNN의 Feature map 분석 - 직접 구현한 실험 결과 공유



- convolution, pooling을 반복하다보면, 점점 더 넓은 범위, 점점 더 많은 특징을 대표하게 됨
- low level feature: 좌-우 에지, 상-하 에지, 대각선 에지, 재질 등등
- middle level feature: 다운 샘플링된 내에서의 그런 것들. 상대적으로 큰 구조물에 대한 정보임
- high level feature: 더 다운 샘플링된 내에서의 그런 것들. 상대적으로 거의 머리, 다리 수준의 표지임.
- 신기한 점: 성공한 그림은 max pooling을 많이한 레이어에서, weight들을 싹 다 더해보니, 실제 물체가 있는 위치에 주목했다. 실패한 그림은 그 외의 배경에 주목했다.
- 주의: 이렇게 훈련된다는 것이 아니라, 이런식으로도 해석할 수 있다 정도
  - 미분해서 반대 방향으로 갔더니, 그렇게 나왔을 뿐이지, 뭘 뽑아냈는지 인간도 머신도 모름







## 06 VGGnet 모델읽기

- VGG는 이미지 그룹 이름 University of Oxford, Visual Geometry Group
- VGGNet은 **옥스포드 대학의 연구팀에 의해 개발된 모델**로써, 2014년 ILSVRC에서 준우승한 모델입니다. 이 모델은 이전에 혁신적으로 평가받던 AlexNet이 나온지 2년만에 다시 한 번 오차율 면에서 큰 발전을 보여줬습니다.

![img](https://blog.kakaocdn.net/dn/b7eZ7d/btqKPVnRGHr/uHmdLNeKQhukEZeGLKgCv1/img.png)

- VGGNet의 종류

  ![img](https://blog.kakaocdn.net/dn/bw2WnC/btqKSgFqB3D/hYW6yfkjGFXmzMMwEM7tZ1/img.png)













