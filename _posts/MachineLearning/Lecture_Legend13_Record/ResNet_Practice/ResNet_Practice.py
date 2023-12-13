import torch
from torch import nn
from torchinfo import summary
from torchviz import make_dot # draw schematic
import os # save directory

class BasicBlock(nn.Module):
    """
    Basic building block for ResNet.
    
    Args:
        in_channels (int): Number of input channels.
        inner_channels (int): Number of inner channels.
        stride (int): Convolution stride.
        projection (nn.Module, optional): Projection layer for shortcut connection.
    """
    expansion = 1  # 확장(expansion) 비율을 1로 설정합니다. layer 아웃이 basic block에서는 input에 비해서 숫자가 변경되지 않았고, bottleneck에서는 균일하게 4배 늘어났다

    def __init__(self, in_channels, inner_channels, stride=1, projection=None):
        """
        Initializes a BasicBlock instance.

        Args:
            in_channels (int): Number of input channels.
            inner_channels (int): Number of inner channels.
            stride (int, optional): Convolution stride. Default is 1.
            projection (nn.Module, optional): Projection layer for shortcut connection. Default is None.
        """
        print("This is Basic, Init")
        super().__init__()  # nn.Module 클래스를 상속하여 BasicBlock 클래스를 생성합니다

        # Residual 경로를 정의합니다.
        # 3x3 두번 주는 것
        self.residual = nn.Sequential(
            # conv_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            # 3x3 컨볼루션 연산을 수행합니다. 입력 채널 수는 in_channels이고 출력 채널 수는 inner_channels입니다.
                # nn.Conv2d 레이어를 생성합니다.
                # in_channels: 입력 채널의 수 (예: RGB 이미지의 경우 3 채널)
                # out_channels: 출력 채널의 수 (필터 수 및 특징 맵의 깊이를 결정)
                # kernel_size: 컨볼루션 커널의 크기 (정수 또는 (높이, 너비)의 튜플)
                # stride: 커널의 이동 간격 (정수 또는 (높이, 너비)의 튜플, 기본값은 1)
                # padding: 입력 주변에 추가할 제로 패딩의 양 (정수 또는 (높이, 너비)의 튜플, 기본값은 0)
                # bias: 레이어에 편향(bias)을 추가할지 여부 (기본값은 True). Batch Normalization과 함께 사용할 때 편향을 사용하지 않는 것이 일반적
                # dilation: 딜레이션(dilation)을 적용할 경우 사용됩니다 (기본값은 1)
                # groups: 입력 채널을 그룹화하는 데 사용됩니다 (기본값은 1)
                # padding_mode: 패딩 모드를 설정합니다 ('zeros', 'reflect', 'replicate', 'circular' 중 하나, 기본값은 'zeros')

            nn.Conv2d(in_channels, inner_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # Batch Normalization 레이어를 추가합니다.
                # 배치 정규화 레이어를 추가하여 출력 채널을 정규화합니다. 이는 학습을 안정화하고 모델 성능을 향상시키는 데 도움이 됩니다.
            nn.BatchNorm2d(inner_channels),
            # ReLU 활성화 함수를 추가하여 비선형성을 적용합니다.
                # inplace=True로 설정하면 함수가 입력 텐서 자체를 수정하고 반환하지 않습니다. 이렇게 하면 메모리 사용량을 줄일 수 있으며 속도를 약간 향상시킬 수 있습니다.
                #기본적으로 inplace=False로 설정되어 있으며, 이 경우 ReLU 함수는 입력 텐서를 수정하지 않고 새로운 텐서를 반환합니다. 
            nn.ReLU(inplace=True),
            # 다시 3x3 컨볼루션 연산을 수행합니다. 출력 채널 수는 inner_channels * self.expansion으로 확장됩니다.
                # inner_channels에서 inner_channels * expansion으로의 두 번째 합성곱 레이어를 정의합니다.
                # 여기서 expansion은 BasicBlock 클래스의 클래스 변수로 설정됩니다.
                # inplace: 원래 위치(데이터)에 직접 적용. 새로 return 받으려면 inplace=False
            nn.Conv2d(inner_channels, inner_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            # 두 번째 배치 정규화 레이어를 추가합니다.
            nn.BatchNorm2d(inner_channels) # inner_channels * self.expansion이어야 한다. 여기선 self.expansion이 1이므로 동일한 결과.
        )

        # 강의에서는 downsample이 있었으나 배포된 코드에는 없어지고 projection이 생겼다
        self.projection = projection
        # 선택적 투영 레이어를 설정합니다.
            # 파이토치(PyTorch)의 nn.Module 또는 이를 상속한 클래스의 인스턴스
            # 투영 레이어(projection layer)와 일반 합성곱 레이어(convolutional layer)는 주로 컨볼루션 신경망(또는 CNN) 내에서 사용되는 다른 유형의 레이어입니다.
            # 투영 레이어 (Projection Layer):
                # 투영 레이어는 입력의 공간 크기와 채널 수를 조절하는 역할을 합니다.
                # 주로 스트라이드(stride)가 2 이상인 합성곱 레이어 뒤에 배치됩니다.
                # 특히 ResNet과 같은 네트워크 아키텍처에서 사용됩니다.
                # 스트라이드가 2 이상인 합성곱 레이어를 사용할 때, 공간 크기가 축소되고 채널 수가 변경되는 문제를 해결하기 위해 투영 레이어가 필요합니다.
            # 일반 합성곱 레이어 (Convolutional Layer):
                # 일반 합성곱 레이어는 주로 입력과 출력의 차원을 변경하지 않고 정보를 전달하는 역할을 합니다.
                # 주로 특징 맵의 깊이(depth)를 변경하거나 공간 크기를 유지하면서 특징을 추출하는 데 사용됩니다.
                # 입력과 출력의 차원을 일치시키지 않고, 정보를 그대로 전달합니다.
                # 컨볼루션 신경망의 중간 부분에서 주로 사용됩니다.
            # 간단히 말하면, 투영 레이어는 입력과 출력의 차원을 조절하는 역할을 하고, 일반 합성곱 레이어는 차원을 변경하지 않고 정보를 전달하는 역할을 합니다. 이러한 레이어는 네트워크 아키텍처와 작업에 따라 선택적으로 사용됩니다.
        self.relu = nn.ReLU(inplace=True)  # ReLU 활성화 함수를 설정합니다.
        

    def forward(self, x):
        """
        Forward pass of the BasicBlock. BasicBlock 생성 시 자동 실행 됨. pytorch nn.Module 특성

        Args:
            x (torch.Tensor): Input tensor. 
            images batch, 배치크기 * 채널 수 * 높이 * 너비

        Returns:
            torch.Tensor: Output tensor.
        """
        # print("This is Basic, forward")
        residual = self.residual(x)
        # 입력 이미지를 신경망에 연결한 것
        # __init__
        # residual: nn.sequential
        

        # 필요한 경우 투영 레이어를 사용하여 shortcut 연결을 수행합니다.
        if self.projection is not None:
            shortcut = self.projection(x)
            print("this is Basic, forward, projection is not None", type(self.projection(x)))
        else:
            shortcut = x
            print("This is Basic, forward, projection is None")

        out = self.relu(residual + shortcut)  # Residual과 shortcut을 더하고 ReLU를 적용합니다.
        return out

class Bottleneck(nn.Module):
    """
    Bottleneck building block for ResNet.
    
    Args:
        in_channels (int): Number of input channels.
        inner_channels (int): Number of inner channels.
        stride (int): Convolution stride.
        projection (nn.Module, optional): Projection layer for shortcut connection.
    """
    expansion = 4

    def __init__(self, in_channels, inner_channels, stride=1, projection=None):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels * self.expansion, 1, bias=False),
            nn.BatchNorm2d(inner_channels * self.expansion)
        )
        self.projection = projection
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x) 

        if self.projection is not None:
            shortcut = self.projection(x)
        else:
            shortcut = x

        out = self.relu(residual + shortcut)
        return out

class ResNet(nn.Module):
    """
    ResNet model.
    
    Args:
        block (nn.Module): Type of building block (BasicBlock or Bottleneck).
        num_block_list (list): List specifying the number of blocks in each layer.
        num_classes (int): Number of output classes.
        zero_init_residual (bool): Whether to zero-initialize the last BN in each residual branch.
    
        ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    """
    def __init__(self, block, num_block_list, num_classes=1000, zero_init_residual=True):
        # ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
        print("This is in ResNet __init__, block type: ", type(block), block)
        super().__init__()

        self.in_channels = 64 # 모델의 입력 채널 수 = 64
        # Conv1 Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # 3: 입력 채널 수, RGB 3
            # 64: 출력 채널 수
            # kernel_size=7: 컨벌루션 커널 사이즈 7x7
            # stride=2: 이동간격, 2픽셀씩 이동
            # padding=3: 컨벌루션 연산 후 입력 이미지 주변에 빈 공간을 추가하여 연산 후 특징 맵의 크기를 유지 함. 3 픽셀의 패딩 추가
            # bias=False: 바이어스 사용하지 않음. weight만 사용
        self.bn1 = nn.BatchNorm2d(64) # 모델의 입력 채널 수와 일치
        self.relu = nn.ReLU(inplace=True) # inplace=True: 입력 텐서 자체를 수정
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # max pooling
            # kernel_size: 윈도우 크기 3x3
            # stride=2: 건너뛰기 2
            # padding=1: 입력 데이터 주위에 추가되는 가상의 픽셀, 입력과 출력의 크기를 조절
        # Layer1-4
        self.layer1 = self.make_layers(block, 64, num_block_list[0], stride=1)
        self.layer2 = self.make_layers(block, 128, num_block_list[1], stride=2)
        self.layer3 = self.make_layers(block, 256, num_block_list[2], stride=2)
        self.layer4 = self.make_layers(block, 512, num_block_list[3], stride=2)
        # Average Pooling and Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # adaptive average pooling, 2d: 
                # 입력 데이터의 공간 차원을 조절하기 위해 필요한 크기(튜플, (1, 1) = 1x1)로 조정
                # 출력으로 1x1 크기의 특징 맵을 생성하도록 지정하는 것입니다. 이렇게 하면 입력 데이터의 공간 차원이 1x1로 축소되고, 각 채널의 평균값이 출력
                # 인풋 데이터를 섹션으로 나눈 후, 섹션의 평균 값을 출력으로 사용
        self.fc = nn.Linear(512 * block.expansion, num_classes)
            # fully connected layer. 최종 출력은 num_classes와 연결되어 분류의 최종 단계임.
            # BasicBlock 혹은 bottleneck의 expansion

        for m in self.modules():
            # print(f"m in self.modules(): {m}")
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu") # He 초기화를 통하여 convolution layer의 weight를 초기화한다.

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.residual[-1].weight, 0)
                    # 모든 layer를 검사하면서, 주어진 block의 마지막 모듈(batchnorm)만 0으로 초기화한다.
                    # batchnorm을 통해서 평균, 분산을 조절할 때에, 분산을 0으로 바꾸는 것
                        # 스케일(γ): 정규화된 입력 데이터에 곱해지는 파라미터
                        # 시프트(β): 정규화된 입력 데이터에 더해지는 파라미터
                        # 가중치를 0으로 초기화하면, 스케일 파라미터가 0이되어 정규화된 입력 데이터에 영향을 미치지 않게 됨. 따라서 BatchNorm 레이어가 학습 초기에는 입력 데이터를 변형시키지 않고 그대로 전달하게 됨.
                        # BatchNorm(X)=γ⋅( (X−μ)/sqrt(σ^2+ϵ) )+β

    def make_layers(self, block, inner_channels, num_blocks, stride=1):
        """
        Create a sequence of blocks for a layer.

        Args:
            block (nn.Module): Type of building block (BasicBlock or Bottleneck).
            inner_channels (int): Number of channels inside the block.
            num_blocks (int): Number of blocks in the layer.
            stride (int): Stride for the convolutional layers.

        Returns:
            nn.Sequential: Sequence of blocks for the layer.
        """
        print(f"This is in make_layers. block: {block}, inner_channels: {inner_channels}, num_blocks: {num_blocks}, stride: {stride}")
        # 만약 입력과 출력의 크기가 다르다면, projection이라는 부분을 사용하여 입력의 크기를 적절하게 변경합니다
        if stride != 1 or self.in_channels != inner_channels * block.expansion:
            #  이 시퀀셜은 먼저 1x1 컨볼루션 레이어로 입력의 크기를 줄이고(nn.Conv2d) 그 다음에 배치 정규화(nn.BatchNorm2d)를 적용합니다. 이를 통해 입력과 출력 크기를 일치시킬 수 있습니다.
            projection = nn.Sequential(
                nn.Conv2d(self.in_channels, inner_channels * block.expansion, 1, stride=stride, bias=False),
                # 입력 채널 수: 3, 출력 채널 수: 64, 커널 크기: 1x1
                # 스트라이드: 2, 패딩: 1, 패딩 모드: "valid", 편향 사용 여부: False
                # conv_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), padding_mode='valid', bias=False)
                nn.BatchNorm2d(inner_channels * block.expansion) # 점섬 커넥션
            )
            
        else:
            projection = None
            #  stride가 1이거나 self.in_channels와 inner_channels * block.expansion이 같으면 projection은 None으로 설정됩니다. 
            # 이는 입력과 출력의 크기가 동일하므로 따로 변경할 필요가 없음을 의미합니다.

        layers = []
        layers += [block(self.in_channels, inner_channels, stride, projection)] # 객체의 참조를 저장하므로, 인스턴스를 직접 리스트에 넣을 수 있다.
        print(f"layers: {layers}")
        self.in_channels = inner_channels * block.expansion
        for _ in range(1, num_blocks):
            layers += [block(self.in_channels, inner_channels)]
        print(f"layers after for: {layers}")
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(num_classes=1000):
    """
    Create a ResNet-18 model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: ResNet-18 model.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes) # BasicBlock을 여기서 초기화하고 동시에 인풋으로 넘겨준다

def resnet34(num_classes=1000):
    """
    Create a ResNet-34 model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: ResNet-34 model.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes) 
    # BasicBlock 클래스 자체를 함수에 전달. 혹은 클래스 레퍼런스 전달이라고 부름.
    # 클래스 자체를 전달하는 경우:
        # 장점: 코드가 간결하고 모델의 구조가 명확하게 드러납니다.
        # 단점: 모델의 상태가 호출 간에 공유되므로 주의가 필요합니다.
    # 클래스 인스턴스를 생성하고 전달하는 경우:
        # 장점: 호출 간에 독립성이 보장되며 각 호출은 자체 모델 인스턴스를 사용합니다.
        # 단점: 모델 인스턴스를 생성하는 추가 오버헤드가 있을 수 있습니다.


def resnet50(num_classes=1000):
    """
    Create a ResNet-50 model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: ResNet-50 model.
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    """
    Create a ResNet-101 model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: ResNet-101 model.
    """
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    """
    Create a ResNet-152 model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: ResNet-152 model.
    """
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


if __name__ == "__main__":
    # model = resnet152()
    print("This is init of resnet18")
    model = resnet18() # Basic
    # model = resnet50() # Bottleneck
    print("This is print model")
    print(model)
    # summary(model, input_size=(2,3,224,224), device='cpu')
    
    input_data = torch.randn(1, 3, 224, 224)  # 예제 입력 형태 (batch_size, channels, height, width)
    
    print("This is model(input_data)")
    output = model(input_data)
    
    dot = make_dot(output, params=dict(model.named_parameters()))
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # dot.render(os.path.join(current_directory, "resnet18_graph2"), format='png', engine='neato')  # "resnet18_graph.pdf"로 저장됩니다.
    # dot.render(os.path.join(current_directory, "resnet18_graph_fdp"), format='png', engine='fdp')  
    # dot.render(os.path.join(current_directory, "resnet18_graph_sfdp"), format='png', engine='sfdp')  
    # dot.render(os.path.join(current_directory, "resnet18_graph_twopi"), format='png', engine='twopi ')  
    # dot.attr(rankdir='LR')  # 좌우 방향으로 정렬
    # dot.attr(rankdir='UD')  # 좌우 방향으로 정렬
    # dot.attr(splines='ortho') # 화살표 사선 -> 직각
    # dot.render('graph')
    # dot.view()  # 기본 PDF 뷰어로 열립니다.
    
