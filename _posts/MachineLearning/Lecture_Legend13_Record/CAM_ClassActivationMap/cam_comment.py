import torch
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
import urllib.request
import ast
import numpy as np
import cv2

## 이미지 경로 설정
img_path = r'D:\GitHub_Project\koyumi0601.github.io\_posts\MachineLearning\Lecture_Legend13_Record\CAM_ClassActivationMap\cat.jpg'  # 분석할 이미지 파일 경로

## Resnet은 ImageNet에서 Training 되었으므로 image Net의 class 정보를 가져옵니다.
classes_url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'

## class 정보 불러오기
with urllib.request.urlopen(classes_url) as handler:
    data = handler.read().decode()
    classes = ast.literal_eval(data)

## Resnet 불러오기
model_ft = models.resnet18(pretrained=True) # resnet을 불러오고 pretrained weight를 가져와서 초기값으로 사용 한다 true
# ResNet은 끝에 GAP - fc로 이어지는 구조이므로, 특별히 네트워크를 뜯어 고치지 않아도 특성맵을 만들어낼 수 있다.
model_ft.eval() 
## model_ft.eval(): 추론모드. 드롭아웃 및 배치정규화 비활성화, 모델 가중치 및 파라미터 고정 - 학습되지 않음.
## model_ft.train(): 학습모드


# print(model_ft)
# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=512, out_features=1000, bias=True)
# )

## Imagenet Transformation 참조
## https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
preprocess = transforms.Compose([
    ## Resize는 사용하지 않고 원본을 추출
   transforms.Resize((224,224)), # 픽셀 크기 224x224로 변환
   transforms.ToTensor(), # 텐서로 변환
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 모델 학습 당시의 평균, 표준편차로 변경. normalized_value = (org_value - mean) / std
])

## 그림을 불러옵니다.
raw_img = Image.open(img_path) # type(raw_img): <class 'PIL.JpegImagePlugin.JpegImageFile'>, raw_img.size: (5184, 3456)

## 이미지를 전처리 및 변형
img_input = preprocess(raw_img) # type: torch.Tensor, size: [3, 224, 224]

# # tensor [채널, 높이, 너비], matplotlib [높이, 너비, 채널]
# plt.imshow(img_input.permute(1, 2, 0)) 
# plt.show()


## 모델 결과 추출
output = model_ft(img_input.unsqueeze(0)) 
# img_input.unsqueeze(0): 배치 차원 추가. [3, 244, 244] -> [1, 3, 244, 244]
# output: tensor. 각 클래스에 대한 로짓(스코어). 로짓(logit) = ln(확률값 / (1 - 확률값))

## 클래스 추출
softmaxValue = F.softmax(output) 
# softmax(logit): 각 클래스에 속할 확률. 
# type: <class 'torch.Tensor'>, size: torch.Size([1, 1000])
# p(c) = exp(logit(c)) / Σ(exp(logit(k))) for k in all classes
# p(c): 클래스 c에 속할 확률
# logit(c): 클래스 c에 대한 로짓 값
# Σ(exp(logit(k)))는 모든 클래스 k에 대한 exp(logit(k)) 값의 합

# plt.plot(softmaxValue.tolist()[0]) # [[numbers]]
# plt.show()

class_id=int(softmaxValue.argmax().numpy())
# softmaxValue.argmax(): torch.tensor에서 최대 값을 가지는 배열의 인덱스를 반환
# print(f'class_id: {class_id}')

## Resnet 구조 참고
## https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def get_activation_info(self, x):
    # See note [TorchScript super()]
    # forward propagation
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    #   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1)): not used
    #   (fc): Linear(in_features=512, out_features=1000, bias=True): not used
    return x

## Feature Map 추출
feature_maps = get_activation_info(model_ft, img_input.unsqueeze(0)).squeeze().detach().numpy()
# get_activation_info(model_ft, img_input.unsqueeze(0)): layer4까지의 출력을 뽑음. 이게 feature map임.
# torch.Size([1, 512, 7, 7]): 데이터 1개, 피쳐맵 512개, 공간 7x7.
# .squeeze(): 크기 1인 차원을 없앰. [1, 512, 7, 7] -> [512, 7, 7]
# .detach(): 그라디언트 떼냄. 계산에 필요 없으므로.
# .numpy(): numpy 배열로 변환.
# .detach().numpy() 한 세트로 쓰임. torch.tensor -> numpy 배열로 변환할 때.


## Weights 추출
activation_weights = list(model_ft.parameters())[-2].data.numpy() # 끝에서 두번째 parameter 행렬. 즉, fc의 weight. 참고로 마지막 행렬은 bias임. 
# print(f'activation_weights size: {activation_weights.shape}') # torch.size([1000, 512])
# for name, param in model_ft.named_parameters():
#     print(f'name: {name}, weight shape: {param.shape}')
# ...
# name: layer4.1.conv2.weight, weight shape: torch.Size([512, 512, 3, 3])
# name: layer4.1.bn2.weight, weight shape: torch.Size([512])
# name: layer4.1.bn2.bias, weight shape: torch.Size([512])
# name: fc.weight, weight shape: torch.Size([1000, 512]) # fc의 weight를 가져옴.
# name: fc.bias, weight shape: torch.Size([1000])    









## numpy로 이미지 변경
numpy_img = np.asarray(raw_img) # PIL 객체를 NumPy 객체로 변환

def show_CAM(numpy_img, feature_maps, activation_weights, classes, class_id):
    ## CAM 추출
    cam_img = np.matmul(activation_weights[class_id], feature_maps.reshape(feature_maps.shape[0], -1)).reshape(feature_maps.shape[1:])
    # # np.matmul: 행렬 곱. 내적.
    # # print(activation_weights[class_id].shape) 
    # # activation_weights: fc layer의 weight. 이 중에서 추론 결과인 tebby cat의 class_id에 해당하는 weight만 뽑아온다. 512x1
    # # feature_maps.reshape(feature_maps.shape[0], -1)).reshape(feature_maps.shape[1:]
    # print(f'feature_maps.shape: {feature_maps.shape}') # (512, 7, 7)
    # print(f'feature_maps.reshape(feature_maps.shape[0], -1).shape: {feature_maps.reshape(feature_maps.shape[0], -1).shape}') 
    # # reshape(512, -1) : 512 제외 나머지 평탄화. 7x7 -> 49
    # print(f'feature_maps.shape[1:]: {feature_maps.shape[1:]}') #  (7, 7)
    # print(f'cam_img.shape: {cam_img.shape}') # 7x7. 
    # # featuremap들의 weighted sum. 그 가중치는 class 판별 직전의 fc layer 출력으로, 결정에 영향을 준 정도라고 볼 수 있다. 영향력.


    # normalize 이후 255로 rescaling
    cam_img = cam_img - np.min(cam_img)
    cam_img = cam_img/np.max(cam_img)
    cam_img = np.uint8(255 * cam_img)
    
    ## Heat Map으로 변경
    heatmap = cv2.applyColorMap(cv2.resize(255-cam_img, (numpy_img.shape[1], numpy_img.shape[0])), cv2.COLORMAP_JET) 
    # # 255-cam_img: 색반전 for heatmap 처리.
    # # COLORMAP_JET는 높은 값이 파란색이라 heatmap을 그리려면 반전해서 적용한다.
    # print(f'numpy_img.shape[1]: {numpy_img.shape[1]}, numpy_img.shape[0]: {numpy_img.shape[0]}') # 5184, 3456

    ## 합치기
    result = numpy_img * 0.5 + heatmap * 0.3 # blend
    result = np.uint8(result)
    
    fig=plt.figure(figsize=(16, 8))
    ## 원본 이미지
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(numpy_img)
    ## CAM 이미지
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(result)
    plt.suptitle("[{}] CAM Image".format(classes[class_id]), fontsize=30)
    plt.show()

show_CAM(numpy_img, feature_maps, activation_weights, classes, class_id)