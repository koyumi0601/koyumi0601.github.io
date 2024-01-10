---
layout: single
title: "ResNet - loss landscape"
categories: machinelearning
tags: [ML, Machine Learning, AI, Legend13]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

*Legend 13 (Image)  Voice Transciption*



# Loss Landscape

#### 중요성

- **최적화**: 최적화 알고리즘의 작동 원리와 목표 지점을 이해하는 데 중요.
- **모델 이해**: 모델의 학습 가능성과 복잡성 파악.
- **과적합**: 과적합 지점을 식별하여 모델의 성능 개선.
- **일반화**: 부드러운 손실 풍경을 가진 모델이 더 좋은 일반화 경향.

#### 특징

- **전역 최소값**: 모델의 최적 성능을 나타내는 지점.
- **지역 최소값**: 전역 최소값이 아닌, 주변보다 낮은 지점.
- **안장점**: 일부 방향에서 최소, 다른 방향에서 최대가 되는 지점.
- **평탄한 지역**: 손실이 거의 변하지 않는 지역, 최적화 속도 저하 가능성.

### 시각화 방법

- **차원 축소**: 고차원 파라미터 공간을 2D 또는 3D로 변환하는 기법. PCA, t-SNE 등의 방법 사용.
- **손실 곡선 그리기**: 특정 방향에 대한 손실 값 계산을 통한 1D 손실 곡선 생성. 모델 최적화 경로와 손실 평탄성 비교 분석.
- **정규화된 방향 선택**: 모델 파라미터에 대한 무작위 방향 선택 및 필터 레벨에서의 정규화. 가중치 차원과 동일한 방향 사용.
- **2D 등고선 플롯**: 두 개의 무작위 방향 선택 및 정규화를 통한 2D 손실 등고선 그리기. 손실 표면의 지형 시각화.
- **3D 손실 표면 시각화**: 2D 등고선 플롯을 기반으로 한 3D 손실 표면 생성. ParaView와 같은 고급 렌더링 도구 활용.
- **학습 과정 동적 시각화**: 학습 과정 중 손실 풍경 변화 추적. 모델의 손실 공간 탐색 관찰.
- **도구 및 라이브러리 활용**: PyTorch, TensorFlow와 같은 머신러닝 프레임워크와 Matplotlib, Plotly, TensorBoard 등의 시각화 도구 활용.
- **연구 및 분석**: 시각화를 통한 최적화 전략, 배치 크기, 학습률의 모델 손실 공간 영향 분석. 모델의 일반화 능력 및 최적화 난이도 평가.





# Paper 

- Visualizing the Loss Landscape of Neural Nets [https://arxiv.org/pdf/1712.09913.pdf](https://arxiv.org/pdf/1712.09913.pdf)

- code [https://github.com/tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape)

- Paper summary
  - "Visualizing the Loss Landscape of Neural Nets"라는 논문은 신경망의 손실 표면을 시각화하는 방법론에 대해 설명합니다. 이 논문에서는 손실 표면을 시각화하기 위한 여러 기술을 제시하며, 이를 통해 신경망의 최적화 과정을 이해하고, 모델의 일반화 능력과 관련된 통찰을 얻을 수 있습니다. 특히, 이 논문은 손실 표면의 복잡성과 최적화 경로를 시각적으로 분석하는 데 중점을 둡니다.
  - **필터별 정규화**
    - 소개: 손실 함수를 플로팅하기 위해 무작위 가우시안 방향 벡터 사용.
    - 목적: 신경망 매개변수의 스케일에 맞게 벡터들을 정규화하여 손실 함수의 곡률을 시각화.
    - 중요성: 다른 손실 함수 간 의미 있는 비교 가능.
  - **1차원 선형 보간**
    - 방법: 두 세트의 매개변수를 선으로 연결하고, 해당 선을 따라 손실 값을 플로팅.
    - 활용: 최소값의 날카로움과 평탄함 연구에 사용.
  - **등고선 플롯 & 무작위 방향**
    - 구현: 중심점과 두 방향 벡터를 선택하여 손실 함수를 1차원 또는 2차원에서 플로팅.
    - 응용: 다양한 최소화 방법의 궤적 탐색 및 최적화기 비교.
  - **날카로움 대 평탄함**
    - 논의: 날카로운 최소화기가 평탄한 것보다 일반화가 더 잘되는지에 대한 논쟁.
    - 관찰: 필터 정규화를 통한 시각화가 일반화 오류와 상관관계가 있음을 제안.
  - **스케일 불변성**
    - 문제: ReLU 비선형성과 배치 정규화 사용 시 신경망의 스케일 불변성 문제 발생.
    - 강조: 적절한 정규화 없이는 손실 함수 플롯 간 비교가 오해를 불러일으킬 수 있음.
  - **일반화와 네트워크 구조**
    - 탐구: 스킵 연결과 같은 네트워크 구조가 손실 풍경과 일반화에 미치는 영향.
    - 중요성: 손실 풍경의 시각화가 신경망의 행동과 모델의 학습 가능성 및 일반화에 미치는 영향 이해에 중요.
  - 요약
    - 이 문서는 손실 풍경의 시각화가 신경망의 성능과 구조적 특성을 이해하는 데 중요한 역할을 한다고 강조합니다. 특히, 필터별 정규화와 같은 방법론은 손실 함수의 곡률을 시각화하고, 다양한 최적화기의 특성을 비교하는 데 유용합니다. 또한, 스케일 불변성과 네트워크 구조가 일반화에 미치는 영향을 분석함으로써, 모델의 학습과 일반화 능력을 향상시키는 방향으로 연구를 진행할 수 있습니다.



# Loss Landscape의 시각화 방법

## 라이브러리와 예제

- Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

param_vals, loss_vals = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(0, 1, 100))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(param_vals, loss_vals, np.sin(param_vals) + np.cos(loss_vals))
ax.set_xlabel('Parameter')
ax.set_ylabel('Loss')
ax.set_zlabel('Loss Value')
plt.show()
```



- Plotly

```python
import plotly.graph_objs as go
import numpy as np

# 가정: loss_vals와 param_vals는 위와 같습니다.
param_vals, loss_vals = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(0, 1, 100))
loss_surface = np.sin(param_vals) + np.cos(loss_vals)

fig = go.Figure(data=[go.Surface(z=loss_surface, x=param_vals, y=loss_vals)])
fig.update_layout(title='Loss Landscape', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
```



- TensorBoard

```python
import tensorflow as tf

# 모델과 로거 설정
model = ... # 모델 정의
log_dir = ... # 로그 디렉토리 경로
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 모델 훈련
model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
```



- PyTorch

```python
# PyHessian
# 이 코드는 모델의 Hessian 행렬의 고유값을 계산합니다
import torch
import torchvision.models as models
from pyhessian import hessian # pyhessian 라이브러리를 불러옵니다.

# 모델 정의 (예: ResNet18)
model = models.resnet18(pretrained=True)

# 데이터셋 정의 (예: CIFAR10)
# CIFAR10 데이터셋을 불러오고 DataLoader를 설정합니다.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)

# 손실 함수 정의
criterion = torch.nn.CrossEntropyLoss()

# Hessian 계산을 위한 데이터셋과 모델, 손실 함수를 hessian 클래스에 전달합니다.
hessian_comp = hessian(model, criterion, dataloader=trainloader, cuda=True)

# 고유값 계산
eigenvalues, _ = hessian_comp.eigenvalues()
print(f'The eigenvalues of the Hessian are: {eigenvalues}')
# 고유값이 크면 손실 표면이 더 가파르고, 작으면 더 평평함을 의미
```



# Paper's code

https://github.com/tomgoldstein/loss-landscape

### 환경 설정

다음 소프트웨어와 라이브러리가 설치된 멀티 GPU 노드를 준비하세요:

- PyTorch 0.4
- OpenMPI 3.1.2
- mpi4py 2.0.0
- numpy 1.15.1
- h5py 2.7.0
- matplotlib 2.0.2
- scipy 0.19

### 사전 훈련된 모델 다운로드

CIFAR-10 데이터셋에 대한 사전 훈련된 모델을 제공된 링크에서 다운로드하고 `cifar10/trained_nets` 디렉토리에 위치시킵니다.

### 데이터 전처리

다운로드한 모델을 훈련시킬 때 사용된 데이터 전처리 방법과 일치하는지 확인하세요.

### 1D 손실 곡선 시각화

#### 1D 선형 보간 생성

1D 선형 보간 방법은 동일한 네트워크 손실 함수의 두 최소화자 사이의 방향으로 손실 값을 평가합니다. 이 방법은 다른 배치 크기로 훈련된 최소화자의 평탄함을 비교하는 데 사용되었습니다. `plot_surface.py` 메소드를 사용하여 1D 선형 보간 그래프를 생성합니다.

```bash
mpirun -n 4 python plot_surface.py --mpi --cuda --model vgg9 --x=-0.5:1.5:401 --dir_type states \
--model_file cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--model_file2 cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=8192_wd=0.0_save_epoch=1/model_300.t7 --plot

```

#### 무작위 정규화된 방향을 따라 플롯 생성

모델 매개변수와 동일한 차원을 가진 무작위 방향을 생성하고 "필터 정규화"를 수행합니다. 그런 다음 이 방향을 따라 손실 값을 샘플링할 수 있습니다.



```bash
mpirun -n 4 python plot_surface.py --mpi --cuda --model vgg9 --x=-1:1:51 \
--model_file cifar10/trained_nets/vgg9_sgd_lr=0.1_bs=128_wd=0.0_save_epoch=1/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --plot

```



### 2D 손실 등고선 시각화

손실 등고선을 그리기 위해 두 개의 무작위 방향을 선택하고 1D 플로팅과 같은 방식으로 정규화합니다.



```bash

mpirun -n 4 python plot_surface.py --mpi --cuda --model resnet56 --x=-1:1:51 --y=-1:1:51 \
--model_file cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot

```



### 3D 손실 표면 시각화

`plot_2D.py`를 사용하여 기본적인 3D 손실 표면 플롯을 생성할 수 있습니다. 더 상세한 렌더링을 원한다면 [ParaView](http://paraview.org/)를 사용하여 손실 표면을 렌더링할 수 있습니다.





# PyHessian

https://cocoa-t.tistory.com/entry/PyHessian-Loss-Landscape-%EC%8B%9C%EA%B0%81%ED%99%94-PyHessian-Neural-Networks-Through-the-Lens-of-the-Hessian












# Transcription
- 클로바노트 https://clovanote.naver.com/

##### 받아쓰기


오케이

그러면 재민 님은요 오케이 내려보겠습니다.

우리 두 분과 함께해 보도록 하겠습니다.

일단 레즈넷을 본격적으로 하기 전에 레즈넷이 어떤 걸 해결했나 그것부터 그 문제가 있거든요.
유명한 문제 그 문제가 뭔지 그것부터 한번 출발해 볼까 합니다.
로스 앤드 스케이라는 건데 로스 앤드 스케이이라는 논문이 나왔었어요.
그래서 이 논문에서 하고 싶은 말은 뭐냐면 이런 겁니다.
일단 로스의 모양을 보려고 하는 게 로스 랜드스케이이고 그야말로 이런 문제가 있어도 돼요.
렐로랑 베치농 같이 쓰면 그라디언트가 충분히 커집니다.
베니싱 그라디언트가 해결이 된 셈이죠. 그런데도 불구하고 너무 기쁘면 학습이 안 되더라라는 거예요.
언더 피팅이 일어나는데 이런 식으로 애매하게 언더패팅이 일어난다.
그러니까 베니싱 그라디언트가 일어나잖아요. 그러면 뭐 이렇게 뜨문뜨문하다가 떨어지지도 않아요.
떨어지지도 않아요. 그냥 맨 위에서 그냥 놀아요.
베니싱 그래터가 일어나면 아예 학습 자체가 안 돼요.
근데 희한하게 한번 그래프 봅시다. 트레이닝 에러 테스트 에러 둘 다 누가 더 못합니까? 학습을 둘 다

깊은 증입니다.

깊은 놈 56층을 가진 녀석이 테스트뿐만 아니고 트레이닝 도 못하더라.
만약에 트레이닝은 잘하는데 테스트 때 못하면 그거는 무슨 문제라고 얘기를 해요 우리가

그렇죠 그거는 오버 피팅인 건데 여기서 나온 문제는 우버 피팅 말고 다른 문제다라는 겁니다.

원인은 못 밝혀냈어요. 저 왼쪽에서 비겨 1 타이밍 성함 이렇게 써 있죠 저 분이 바로 레즈넷 만든 분이고 레즈넷 페이퍼의 피겨 원이에요.
저게 딥 레즈듀얼 러닝 이렇게 써 있잖아요. 저게 레즈넷 논문입니다.
저런 문제를 발견했는데 왜 저런 일이 일어났는지까지는 그 논문에서 밝혀내지는 못했습니다.
그렇지만 나중에 밝혀지기를 이렇게 모양이 좀 꾸불꾸불해지더라라는 거예요.
저기 스키 커넥션 이렇게 써 있잖아요. 레즈넷의 스키 커넥션이 이 꼬불꼬불해지는 문제를 해결해 준 녀석인 건데 오른쪽 아래 논문은 카이번의 논문 아닙니다.
2018년도의 논문이죠. 그러니까 2016년도에 레즈넷이 나왔어요.
그래서 깊은데 이상하게 언더패팅이 일어나는 그 문제를 해결했습니다.
근데 뭐가 문제인지는 못 밝혀냈고 2018년에 돼서야 아마 로스 랜드스케입이 꼬불거리기 때문에 문제가 된 것 같다.
이렇게 한 거예요. 그러니까 현상을 밝힌 거죠. 레이어를 깊게 깊게 쌓였더니 56층쯤 되니까 꼬불꼬불하더라라는 겁니다.
아무래도 꼬불꼬불하면 그렇죠 학습이 어렵겠죠 좀 다른 데 들어갈 수도 있는 거고 이해돼요 무슨 말인지 루스의 모양이 이런 식으로 생겼으면 다른 데 어디 기툰 이런 데가 있을 수도 있는데 너무 꼬불꼬불거리고 있으니까 여기서 출발해서 아담을 한다고 하더라도 이런 데서 빠진다는 거예요.
여기가 로컬 미니멈인 거예요. 여긴 더 좋은 더 좋은 로컬 매니 글로벌 글로벌이 여기 있다 쳐요.
여기를 가고 싶었을 텐데 못 가고 이런 데 빠져서 못 나오더라라는 거.

그래서 이렇게 로스의 모양을 봄으로써 어떤 모델 모델의 학습이 잘 될까 안 될까를 가늠해 보는 것이 가능해진 거예요.
이 논문 덕분에 로스 랜드스케 노무 덕분에 근데 이 레즈넷에서 제안한 스키 커넥션 을 적용했더니 적용하고 로스의 모양을 봤더니 평평해요.
그죠? 여기에서 출발하더라도 이렇게 쭉 들어와서 일로 잘 빠질 수가 있겠다라는 거.

스키 커미션이 뭔지를 우리가 아는 게 레즈넷을 이해하는 것이고 이 로스 엔디스케이에서는 봐라 스키 커넥션이라는 것을 적용했더니 이렇게 모양이 예뻐지지 않느냐라는 당연히 얘보다 얘가 성능이 좋겠죠 딱 봐도 그거는 그렇습니다.
이 모양을 어떻게 그려냈을까 그걸 제안한 게 바로 이 논문인 거예요.
이렇게 그림을 어떻게 그리는지 로스를 우리가 그림을 그릴 수가 있을까요? 3차원에다가 그냥 웨이트의 함수로 그것부터 이은 목까? 그냥은 못 그리죠.
왜죠? 차원이 달라서요. 차원이 수천만 수 100만 차원에 놓이는 거잖아요.
로스라는 게. 이해돼요 무슨 말인지 웨이트가 수백만 개인데 축이 하나면 이렇게 2d로 그림이 그려지고 축이 두 개면 이렇게 3d로 그려질 텐데 축이 몇백만 개인 거예요.
어떻게 그런 그림을 그리냐 바로 얘가 아주 기가 막힌 아이디어를 냈는데 일단 있었던 아이디어가 있어요.
원래 있었던 아이디어는 랜덤 백터 2개를 뽑아라.
웨이트 벡터랑 똑같은 사이즈를 가진 랜덤 벡터 2개를 뽑아라.
그래가지고 알파랑 베타를 마이너스 1에서 1까지 바꿔가면서 그림을 그려라.
이렇게 제안된 바가 있습니다. 근데 그렇게 그림을 그렸더니 좀 이상해져가지고 뭔가 짱구를 굴려가지고 뭔가 처리를 특별한 처리를 한 랜덤 벡터를 가지고 이렇게 스케닝을 하자라는 거예요.
그러니까 뭔 말이냐 세타얘는 로컬 미니멈에 해당되는 웨이트인 거예요.
그 웨이트가 100만 개 있다 쳐요. 그러면 얘의 길이는 어떻게 됩니까?

웨이트 1 웨이트 e 몇 개가 되는 거예요? 웨이트가 100만 개면

10만 개

100만 개 근데 이 웨이트가 cnn의 웨이트라고 합시다.
그러면 여기가 첫 번째 레이어에 첫 번째 필터 그리고 여기가 첫 번째 레이어에 두 번째 필터

저기

뭐 이상한 점 있으세요? 질문 질문 있었어요 아닌가 이렇게 쭉 되겠죠 이해돼요? 무슨 말인지 웨이트가 첫 번째 레이어의 첫 번째 필터 첫 번째 레이어에 두 번째 필터 그리고 마지막 레이어에 마지막 필터 이렇게 웨이트 값이 쭈르르르륵 있을 겁니다.
맞습니까? 이해됐어요? 입자 사이트 한번 들어와서 눌러주시고 노트 협 오케이.
저도 계속 보고 있어요. 오케이 좋습니다. 그런데 짱구를 굴렸다 그랬어요.
어떻게 굴렸냐 이거를 이거를 각각의 축으로 해가지고 그림을 그릴 수가 있다 없다? 없기 때문에 그냥 랜덤한 벡터를 만들어라.
다만 그 길이가 100만 개인 100만에 해당되는 랜덤한 웨이트 벡터를 만들어라라는 거예요.
이렇게 이거 랜덤 값이에요. 그냥 랜덤 값. 이게 100만 개 이게 델타 델타 에타 같은 방식으로 만드는 겁니다.
델타만 생각하시면 돼요. 델타 에타 똑같은 방식으로 만들 겁니다.
그랬을 때 델타는 랜덤한 값으로 하나 벡터를 만든다.
100만 개짜리. 그러고 나서 이 델타랑 에타를 만들고 이 벡터를 축으로 스케일을 시키는 거죠.
여기다가 알파 여기다가 베타 하면은 그러면 말하자면 어떤 로스가 있을 때 100만 차원 위에 놓인 롤스가 있겠죠 이게 100만 차원인 건데 여기 여기가 100만 차원 위에다 이제 그림이 그려지겠죠 거기에 델타 에타가 있는 거고 근데 델타 에타를 바꿔가면서 그림을 그리면 그 바꿔가는 정도를 알파랑 베타로 놓고 3차원 위에 그림을 그리자라는 거예요.
알파랑 베타를 바꿔가면서 l 값을 보자. 대신 그 축 자체는 데이터랑 에타 축인 거예요.
데이터랑 에타 축으로 쭉쭉 해가지고 로스의 모양을 그리자.
알파만큼 가고 델타만큼 가서 로스가 그리고 그럼 알파베타가 0이면 그때는 어떻게 되는 거예요? 로컬 미니멈 값에 해당되는 로스를 보는 거죠.
이해돼요? 무슨 말인지 알파랑 메타가 0이면 됐어요 알파랑 메타가 0이면 그냥 로컬 미니멈 값을 보게 될 것이다.
여기서 관건입니다.

이렇게 옛날에 했었는데 스케일링 문제가 좀 있었대요.
그래가지고 얘네가 어떻게 생각했냐면 여기에 필터 있죠 사이즈가 있을 거잖아요.
첫 번째 필터 사이즈가 거기에 해당되는 똑같은 사이즈를 가진 얘네들을 얘네 크기랑 얘네 크기랑 똑같이 맞춘 거예요.
여기의 크기랑 여기의 크기랑 이해돼요? 무슨 말인지 요 벡터의 크기랑 이 랜덤 벡터의 크기랑 일치하게끔 맞춰준 겁니다.
어떻게 맞춰주냐 여기를 델타아이마 델타 1 1이라고 합시다.
1 1 1 콤마 1 첫 번째 레이아웃 첫 번째 설정이 있다.
이게 벡터예요. 그러면 이 벡터를 일단 노말라이즈를 하는 거예요.
이렇게 투노머로. 그러면 얘는 크기가 몇이에요?

예

1이죠. 여기다가 이게 웨이트 1 콤마 1이라고 하면 1 콤마 1 벡터 그러면 여기다가 이만큼을 곱해주죠.
델타 세타 별 일 콤마일이죠 델타 별 일 콤마일

그러면 얘는 크기가 어떻게 돼요?

등록

싫다. 그거만큼

얘만큼 크기가 스케일이 되겠죠 그 행위를 모든 필터에 대해서 다 해주는 거예요.
그렇게 해서 그 축을 담당하는 베이시스 벡터를 만드는 겁니다.
그러고 나서 알파랑 메타를 바꿔가면서 로스를 그림을 그리면 이렇게 나오더라라는 거예요.
그렇게 해서 로스 레드 스킵을 그려봤고 스키 커넥션의 유무 혹은 레이어가 많아질수록 어떻게 되는지 그걸 본 거예요.
봤더니 노스키 커넥션 스키 커넥션 안 했더니 ns 로스 모양이 이상해지더라.
abc랑 be f를 비교하면 되겠죠 def부터 보겠습니다.
층을 깊게 만드니까 스키 커넥션 없이 깊게 만드니까 어떻게 됐습니까? 루스를 모양이

루스의 모양이 d의 두 순으로 루스의 모양이 찌그러져 찌그러지죠 반면에 abc는 어떻습니까? 예뻐요 이쁘죠 그러니까 누가 더 학습이 쉽겠어요 abc가 학습이 쉬워요 df가 더 쉬워요 abc abc가 훨씬 쉽죠 바로 그겁니다.
그걸 밝혀낸 논문이 바로 이 루스 랜드 스킨 논문이 이걸 가지고 트랜스포머도 그려볼 수 있고 여러 가지 그려볼 수 있겠죠 그런 녀석이 그런 하나의 툴이에요.
로스 랜드스케이블 그릴 수 있는 툴 그겁니다. 스키 커넥션이라는 존재가 저렇게 좋다는 거고 스키 이 뭔지 설명 들어갑니다.
레지 넷.


clovanote.naver.com



# Transcription Summary

방법론

1. **랜덤 벡터 생성**
   - 모델 가중치와 동일한 크기의 두 개의 랜덤 벡터 생성
   - 각 가중치 필터에 맞춰 랜덤 벡터 스케일링 및 정규화
2. **벡터 정규화**
   - 생성된 랜덤 벡터의 크기를 1로 조정하여 방향 유지
3. **스케일링**
   - 정규화된 벡터에 모델의 각 가중치 필터 크기를 곱하여 원래 스케일로 조정
4. **정규화**
   - 벡터의 크기를 표준화하여 방향만을 고려하도록 조정
5. **손실 풍경 시각화의 스케일링과 정규화**
   - 랜덤 벡터를 기저로 사용하여 가중치 공간의 두 방향 표현
   - 가중치 조정을 통한 손실 함수 값 계산 및 2차원 슬라이스 생성
6. **손실 풍경 플롯**
   - 알파와 베타 매개변수를 변화시키며 손실 값을 계산
   - 계산된 손실 값을 3차원 공간에 플롯하여 시각화
7. **로컬 미니멈 탐색**
   - 알파와 베타가 0일 때 현재 모델 위치에서의 손실 값을 확인