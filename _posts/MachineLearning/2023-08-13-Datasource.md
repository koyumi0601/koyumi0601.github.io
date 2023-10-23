---
layout: single
title: "Data Source"
categories: machinelearning
tags: [ML, Machine Learning, Dataset]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

*Data sources*



# AI

- MNIST : 

  - **MNIST (Modified National Institute of Standards and Technology)** 
  - 손글씨 데이터
  - 총 70000개, 60000개 훈련데이터, 10000개 테스트 데이터
  - 28x28 Grayscale (0-255)
  - 레이블 0-9 총 10개의 클래스

  ```python
  import torchvision.transforms as transforms
  from torchvision.datasets import MNIST
  from torch.utils.data import DataLoader
  
  # 데이터 전처리 (예: 텐서로 변환)
  transform = transforms.Compose([transforms.ToTensor()])
  
  # MNIST 데이터셋 다운로드
  mnist_train = MNIST(root='./data', train=True, transform=transform, download=True)
  mnist_test = MNIST(root='./data', train=False, transform=transform, download=True)
  
  # DataLoader를 사용하여 배치 처리
  train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
  test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)
  ```

  

- CIFAR-10 : 

  - Canadian Institute For Advanced Research (후원)
  - 10개의 다른 클래스 
  - 레이블
    - 0 Airplane 비행기
    - 1 Automobile 자동차
    - 2 Bird 새
    - 3 Cat 고양이
    - 4 Deer 사슴
    - 5 Dog 개
    - 6 Frog 개구리
    - 7 Horse 말
    - 8 Ship 배
    - 9 Truck 트럭
  - 60000개, 32x32 컬러 이미지
  - 50000개: 훈련데이터, 나머지 10000개: 테스트 데이터

  ```python
  from torchvision.datasets import CIFAR10
  
  # 데이터 전처리 (예: 텐서로 변환 및 정규화)
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  
  # CIFAR-10 데이터셋 다운로드
  cifar10_train = CIFAR10(root='./data', train=True, transform=transform, download=True)
  cifar10_test = CIFAR10(root='./data', train=False, transform=transform, download=True)
  
  # DataLoader를 사용하여 배치 처리
  train_loader = DataLoader(cifar10_train, batch_size=32, shuffle=True)
  test_loader = DataLoader(cifar10_test, batch_size=32, shuffle=False)
  ```

- STL-10

  - 스탠포드 대학교 제공
  - Stanford Tiled
  - 이미지 분류 벤치마크 데이터셋, CIFAR-10보다 이미지의 해상도가 높고 데이터셋의 구성이 약간 다름
  - 비지도학습 알고리즘의 성능 평가를 위해 사용
  - 96x96 컬러 이미지
  - 클래스 10개: 비행기, 새, 자동차, 고양이, 사슴, 개, 말, 원숭이, 배, 트럭
  - 학습데이터 5000개, 클래스당 500개 / 테스트데이터 8000개
  - 비지도학습을 위한 10만개의 레이블 없는 이미지 포함 

  ```python
  import torchvision.transforms as transforms
  from torchvision.datasets import STL10
  from torch.utils.data import DataLoader
  
  # 데이터 변환 (ToTensor는 이미지를 PyTorch 텐서로 변환합니다)
  transform = transforms.ToTensor()
  
  # STL-10 학습 데이터 다운로드
  train_dataset = STL10(root='./data', split='train', download=True, transform=transform)
  
  # STL-10 테스트 데이터 다운로드
  test_dataset = STL10(root='./data', split='test', download=True, transform=transform)
  
  # 데이터 로더 생성 (예: 배치 크기 32로 설정)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
  ```

  



# Images

- AI Hub [https://aihub.or.kr/](https://aihub.or.kr/)

# Data
- 한국 정부에서 제공하는 공공데이터 [http://data.go.kr](http://data.go.kr)
- 한국 통계청에서 공개하는 데이터 [http://kostat.go.kr](http://kostat.go.kr)
- 한국 보건 의료 빅데이터 개방 시스템 [http://opendata.hira.or.kr](http://opendata.hira.or.kr)
- 한국 지방행정 인허가 데이터 [http://www.localdata.kr](http://www.localdata.kr)
- 한국 문화체육관광부 문화 데이터 [https://www.mcst.go.kr](https://www.mcst.go.kr)
- 서울시 열린데이터 광장 [http://data.seoul.go.kr](http://data.seoul.go.kr)
- 경기도 공공데이터 개방 포털 [https://data.gg.go.kr](https://data.gg.go.kr)
- 미국 정부의 공공데이터 [http://data.gov](http://data.gov)
- 세계 은행에서 제공하는 개방 데이터 [http://data.worldbank.org](http://data.worldbank.org)
- 미국 식약청의 개방 데이터 [http://open.fda.gov](http://open.fda.gov)
- naver [http://naver.com](http://naver.com)
- twitter [https://twitter.com](https://twitter.com)
- meta [https://www.facebook.com](https://www.facebook.com)
- 와인 데이터셋, 어바인 대학 머신러닝 저장소 [https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality)



# R

# Web API 제공자
- 네이버 개발자 센터 [https://developers.naver.com](https://developers.naver.com)
- 카카오 앱 개발 플랫폼 서비스 [https://developers.kakao.com](https://developers.kakao.com)
- 페이스북 개발자 센터 [https://developers.facebook.com](https://developers.facebook.com)
- 트위터 개발자 센터 [https://developers.twitter.com](https://developers.twitter.com)
- 공공데이터포털 [https://www.data.go.kr](https://www.data.go.kr)
- 세계 날씨 [http://openweathermap.org](http://openweathermap.org)
- ~~유료/무료 API 스토어 [http://mashup.or.kr](http://mashup.or.kr)~~
- ~~유료/무료 API 스토어2 [http://www.apistore.co.kr/api/apiList.do](http://www.apistore.co.kr/api/apiList.do)~~