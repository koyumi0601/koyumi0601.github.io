import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models.vgg import vgg19
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 데이터 준비: 저해상도 이미지 로드 또는 생성

# Generator 모델 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Generator 아키텍처 예시
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator 모델 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Discriminator 아키텍처 예시
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.model(x)

# Generator와 Discriminator 모델 생성
generator = Generator()
discriminator = Discriminator()

# 손실 함수 및 최적화 기준 정의
criterion = nn.MSELoss()  # MSE 손실 함수 사용
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# 이미지 업샘플링 함수 정의
def upscale_image(image):
    # 사용할 업샘플링 알고리즘 선택 (예: Bicubic Interpolation)
    interpolation = cv2.INTER_CUBIC
    
    # 이미지 크기를 원하는 크기로 조정
    desired_size = (image.shape[1] * 2, image.shape[0] * 2)
    upscaled_image = cv2.resize(image, desired_size, interpolation=interpolation)
    
    return upscaled_image

# 테스트 이미지 로드 또는 생성
# 여기에서는 테스트 이미지를 로드하는 대신 무작위 노이즈 이미지를 생성하여 사용합니다.
test_image = np.random.rand(100, 100, 3) * 255.0

# 이미지 업샘플링 적용
super_resolution_image = upscale_image(test_image) # 고전의 방법

# 결과 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(test_image.astype(np.uint8))
plt.title("low resolution image")

plt.subplot(1, 2, 2)
plt.imshow(super_resolution_image.astype(np.uint8))
plt.title("high resolution image")

plt.show()


# Need realize paper: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network