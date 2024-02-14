# path
- 메인 코드 경로 (백업, 20240206)
    - /home/ko/Documents/GitHub/koyumi0601.github.io/_posts/ImageSignalProcessing/QT/project_build/Tesseract_OCR_C_CPU
- refactory 코드 경로
    - /home/ko/Documents/GitHub/koyumi0601.github.io/_posts/ImageSignalProcessing/QT/project_build/Tesseract_OCR_C_CPU_refactory
- 메인코드 카피 후 depth 관련 작업

- 데이터 경로
    - ~~./Tesseract_OCR_C /home/ko/Documents/GitHub/koyumi0601.github.io/_posts/ImageSignalProcessing/QT/project_build/Tesseract_OCR_C_CPU/resource/linear_images~~

- 빌드 경로
    - /home/ko/Documents/build-Tesseract_OCR_C_CPU-Desktop_Qt_5_12_12_GCC_64bit-Debug
- 실행 커맨드
    - ./Tesseract_OCR_C ./linear_images/
- 커맨드 창 경로, 유저, pc이름 지우기
    - export PS1="\$ "
- QT > Projects > run > argument 넣을 수 있음

# Requirement
- Refactoring
- ~~Rubber band -> 마우스버튼 (done)~~
    - ~~release 시점에 사각박스 (stencil ROI box size 15)~~
    - ~~클릭좌표 중심으로 생성~~
    - ~~드래그 시 이동가능~~
    - ~~드래그 시 이동은 마우스와 박스 간의 상대경로가 유지 됨~~
    - ~~겹치는 이동은 하지 않는다. 부딪히면 놓은 것과 같은 동작~~
- export struct 
    - 디자인 예정(~2/15, hold)
- ~~OCR box는 system, probe에 대해서 단일해야 함. (done)~~
    - ~~갱신 시 기존 데이터에 대한 유효성 검사(같은 depth가 출력되는 지) 확인 후 통과하면 새로 생성되도록~~
        - ~~invalid: 사각박스 빨간색~~
        - ~~valid: 사각박스 초록색~~
        - ~~review data 갱신~~
- point 6점 structdata에 넘기기

```cpp
float topLeftX;
float topLeftY;
float topRightX;
float topRightY;
float midLeftX; 중앙 수직선의 위
float midLeftY;
float midRightX;
float midRightY;
float bottomLeftX; 중앙 수직선의 아래, wingClip되지 않은 경우 mid와 bottom의 값은 같아야 함.
float bottomLeftY;
float bottomRightX;
float bottomRightY;
```

# Bug
- ~~Generate Stencil ROI 버튼~~
    - ~~3, 4번째 점이 좌우 갭이 크면 hanging -> center of mass에서 위 2개 좌 2개 뽑는 것으로 변경 (done)~~
    - ~~-1이 나오면 cv 크래시 -> 계산 돌지 않도록 수정 (done)~~

- ~~OCR button~~
    - ~~review 상태에서 새로 그리는 경우, select/unselect할 때 서로 mismatch -> livedata 새로 생성된 것에도 적용~~



# FOV 검출
- 이미지 전처리:
    - 이미지 전처리를 통해 원하는 형태의 폐곡선을 더 잘 검출할 수 있도록 이미지를 조작합니다. 예를 들어, 에지 검출 (Edge Detection) 및 이미지 이진화 (Binary Image)를 사용하여 폐곡선의 테두리를 강조할 수 있습니다.
- 에지 검출:
    - Canny, Sobel, Roberts, Prewitt 등과 같은 에지 검출 알고리즘을 사용하여 이미지에서 에지를 감지합니다. 에지는 폐곡선의 외곽을 표시하는데 도움이 됩니다.
- 형태 검출 (Shape Detection):
    - 에지 검출 결과를 사용하여 폐곡선과 유사한 형태를 찾습니다. 폐곡선의 경우, 주어진 에지를 기반으로 직사각형 또는 부채꼴의 꼭지점과 변을 찾아야 합니다.
- 허프 변환 (Hough Transform):
    - 허프 변환을 사용하여 이미지에서 직선 및 원을 검출할 수 있습니다. 폐곡선의 일부는 직선이나 호로 나타날 수 있으므로 허프 변환을 활용하면 이러한 형태를 검출할 수 있습니다.
    - 허프 변환(Hough Transform)은 이미지 처리 및 컴퓨터 비전 분야에서 사용되는 중요한 기술 중 하나로, 주로 직선 및 원과 같은 기하학적 모양을 검출하는 데 사용됩니다. 허프 변환은 이미지에서 모양을 찾기 위해 주어진 점들을 수학적 모델에 부합하는 형태로 변환하는 프로세스를 의미합니다.
    - 허프 변환의 주요 특징 및 원리는 다음과 같습니다:
        - 직선 및 원 검출: 허프 변환은 이미지에서 주어진 점들을 기반으로 직선과 원을 검출하는 데 주로 사용됩니다.
        - 모델 파라미터 공간: 허프 변환은 검출하려는 모양(예: 직선 또는 원)을 표현하는 모델 파라미터 공간을 정의합니다. 직선의 경우, 모델 파라미터는 기울기와 y 절편입니다.
        - 점의 변환: 주어진 이미지에서 각 점은 모델 파라미터 공간에서 하나 이상의 모델 파라미터를 투영합니다. 이러한 투영을 통해 점들이 모델 파라미터 공간에서 교차점을 찾게 됩니다.
        - 투영 결과 해석: 허프 변환은 모델 파라미터 공간에서 교차하는 지점들을 분석하여 주어진 모양(예: 직선 또는 원)의 모델 파라미터를 결정합니다.
        - 임계값 및 검출: 허프 변환은 임계값을 설정하여 어떤 모델 파라미터를 찾을지 결정하며, 검출된 결과를 얻을 수 있습니다.
        - 예를 들어, 직선 검출의 경우, 이미지에서 점들을 허프 변환을 통해 모델 파라미터 공간으로 투영하면, 모든 점들이 모델 파라미터 공간에서 하나의 지점으로 교차하는 경우가 직선으로 판단됩니다. 이를 통해 이미지에서 직선을 검출할 수 있습니다.
- 꼭지점 검출:
    - 꼭지점 검출 알고리즘을 사용하여 이미지에서 직사각형이나 부채꼴의 꼭지점을 찾습니다. OpenCV와 같은 이미지 처리 라이브러리에는 꼭지점 검출을 위한 함수 및 도구가 제공됩니다.
- 검출된 형태 결합:
    - 검출된 꼭지점과 에지 정보를 결합하여 직사각형 또는 부채꼴의 완전한 형태를 복원합니다.
- 결과 시각화:
    - 최종적으로 검출된 폐곡선을 원본 이미지에 표시하거나 원하는 형태의 정보로 출력합니다.


