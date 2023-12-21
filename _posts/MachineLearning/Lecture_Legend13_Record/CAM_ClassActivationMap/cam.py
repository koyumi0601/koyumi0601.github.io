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
model_ft = models.resnet18(pretrained=True)
model_ft.eval()

## Imagenet Transformation 참조
## https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    ## Resize는 사용하지 않고 원본을 추출
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

## 그림을 불러옵니다.
raw_img = Image.open(img_path)

## 이미지를 전처리 및 변형
img_input = preprocess(raw_img)

## 모델 결과 추출
output = model_ft(img_input.unsqueeze(0))

## 클래스 추출
softmaxValue = F.softmax(output)
class_id=int(softmaxValue.argmax().numpy())

## Resnet 구조 참고
## https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def get_activation_info(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x

## Feature Map 추출
feature_maps = get_activation_info(model_ft, img_input.unsqueeze(0)).squeeze().detach().numpy()
## Weights 추출
activation_weights = list(model_ft.parameters())[-2].data.numpy()
## numpy로 이미지 변경
numpy_img = np.asarray(raw_img)

def show_CAM(numpy_img, feature_maps, activation_weights, classes, class_id):
    ## CAM 추출
    cam_img = np.matmul(activation_weights[class_id], feature_maps.reshape(feature_maps.shape[0], -1)).reshape(feature_maps.shape[1:])
    cam_img = cam_img - np.min(cam_img)
    cam_img = cam_img/np.max(cam_img)
    cam_img = np.uint8(255 * cam_img)
    
    ## Heat Map으로 변경
    heatmap = cv2.applyColorMap(cv2.resize(255-cam_img, (numpy_img.shape[1], numpy_img.shape[0])), cv2.COLORMAP_JET)
    
    ## 합치기
    result = numpy_img * 0.5 + heatmap * 0.3
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