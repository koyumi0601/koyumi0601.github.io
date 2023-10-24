import torchvision # Pytorch, computer vision, image transform, pretrained model, dataset (CIFAR-10, MNIST) interface
import torchvision.transforms as transforms # image transform such as image to tensor, normalization, rotation, resize
import pickle # serialize or deserialize python object

# 데이터 전처리
# 여러 변환 함수를 연결하여 하나의 변환으로 구성
# tensor화 + normalize [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR-10 데이터셋 로드
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 데이터 저장
with open('trainset.pkl', 'wb') as f:
    pickle.dump(trainset, f)
with open('testset.pkl', 'wb') as f:
    pickle.dump(testset, f)