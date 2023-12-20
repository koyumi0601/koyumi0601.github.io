import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 이미지 전처리 및 변환 함수 정의
def preprocess_image(image_path):
    # 이미지 불러오기
    image = Image.open(image_path)
    
    # 이미지 크기 조정 (ResNet-50의 기본 입력 크기)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 전처리된 이미지 반환
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    return input_batch

# 새로운 ResNet 모델 생성 (컨볼루션 레이어와 분류 레이어 변경)
model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
model.eval()

# 커스텀 GAP 레이어 정의
class CustomGAP(nn.Module):
    def __init__(self):
        super(CustomGAP, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=[2, 3])

# 커스텀 GAP 레이어 추가
model.avgpool = CustomGAP()

# 이미지 불러오기 및 전처리
image_path = r'D:\GitHub_Project\koyumi0601.github.io\_posts\MachineLearning\Lecture_Legend13_Record\CAM\cat.jpg'  # 분석할 이미지 파일 경로
input_image = preprocess_image(image_path)

# forward 패스 수행
with torch.no_grad():
    features = model(input_image)

# CAM 계산
class_idx = torch.argmax(features)
weight = model.fc.weight[class_idx]
cam = torch.matmul(features, weight.T).squeeze()

# CAM 시각화
cam = cam - cam.min()
cam = cam / cam.max()
cam = cam.numpy()

# 원본 이미지 로드 및 시각화
original_image = Image.open(image_path)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')

# CAM 시각화
plt.subplot(1, 2, 2)
plt.imshow(cam, cmap='jet')
plt.title('Class Activation Map (CAM)')

plt.show()