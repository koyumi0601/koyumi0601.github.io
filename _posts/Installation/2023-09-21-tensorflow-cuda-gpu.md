---
layout: single
title: "How to make tensorflow, pytorch, gpu, cuda"
categories: setup
tags: [Ubuntu, Installation]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true


---

*How to setup environment for Machine Learning*


요약하자면
- 그래픽 드라이버, CUDA toolkit(11.8), ~~cuDNN~~, tensorflow
- 그래픽 드라이버, CUDA toolkit any (12.2 > 11.8), ~~cuDNN~~, pytorch

주의할 점
- tensorflow는 호환성이 아주 중요하다
- 파일 다운로드, 압축 풀기, 경로에 복사해서 위치시키기, 환경 변수에 넣어주기가 모두 잘 되어야 한다. 


cuda 11.8, cuDNN 8.xx, tensorflow 2.13인데 GPU 장치는 인식되지만 모델학습이 안된다
```python
model.fit(train_images, train_labels, epochs=10, batch_size=64)  # 모델을 학습시킵니다. Need cuDNN libraries
```

설치된 그래픽 드라이버, CUDA toolkit version, cuDNN version, pytorch version 정보는 아래와 같이 확인한다.


```bash
nvidia-smi # 그래픽 드라이버
nvcc --version # CUDA
cat cudnn.h | grep CUDNN_MAJOR -A 2 # cuDNN
```
cuDNN은 설치 경로에 복사되어 있는 지 확인한다
컴퓨터 > usr > local > cuda-x.x > include > cudnn*.h

아카이브에서 .deb 파일을 다운받고 dpkg했을 때 /var/cudnn-local-repo-ubuntu2004-8.9.5.29 폴더가 생성되고 그 하위에 또 .deb들이 생성된 것이 보인다. > 원하는 동작이 아니다
다시 아카이브에서, .tar를 받고 압축을 푼 후(우클릭 > 압축풀기), 필요한 경로(cuda toolkit 설치 경로)에 복사/붙여넣기 한다.


# 복사해서 붙여 넣는 방법

압축 푼 폴더 내에서 터미널을 열고

```bash
# cuda 
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.3.28_cuda11-archive/include/cudnn*.h /usr/local/cuda/include/
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.3.28_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64/
# cuda-11
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.3.28_cuda11-archive/include/cudnn*.h /usr/local/cuda-11/include/
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.3.28_cuda11-archive/lib/libcudnn* /usr/local/cuda-11/lib64/
# cuda-11.8
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.3.28_cuda11-archive/include/cudnn*.h /usr/local/cuda-11.8/include/
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.3.28_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64/
# cuda-12
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/include/cudnn*.h /usr/local/cuda-12/include/
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/lib/libcudnn* /usr/local/cuda-12/lib64/
# cuda-12.1
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/include/cudnn*.h /usr/local/cuda-12.1/include/
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/lib/libcudnn* /usr/local/cuda-12.1/lib64/
# cuda-12.2
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/include/cudnn*.h /usr/local/cuda-12.2/include/
sudo cp /home/ko/Downloads/cudnn-linux-x86_64-8.9.4.25_cuda12-archive/lib/libcudnn* /usr/local/cuda-12.2/lib64/

```


환경변수 (path)에 libr64, include를 추가해준다(임시)
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/include:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH # check registered paths
```


터미널을 다시 열면, 도로 안된다. 환경변수 파일을 수정(계속 유지됨)

- nano ~/.bashrc
- .bashrc의 맨 아래 추가한다.

```bash
export PATH=/usr/local/cuda-11.8/bin:/usr/local/cuda-11.8/NsightCompute-2023.1:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/include:$LD_LIBRARY_PATH
```

- 적용한다 

```bash
  source ~/.bashrc
```

- 확인한다

```bash
  echo $LD_LIBRARY_PATH
  echo $PATH
```


- 참고 중인 블로그 [https://normal-engineer.tistory.com/356](https://normal-engineer.tistory.com/356)


- Install graphic driver 

```bash 

```

- Install CUDA
  - Tensorflow requires CUDA Toolkit 11.8 https://developer.nvidia.com/cuda-11-8-0-download-archive

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

기존의 것을 삭제한 후 다시 깔았다면 nvidia-smi에서 library mismatch 오류가 날 수 있다. 재부팅하면 된다.

- CUDA toolkit release note [https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#)


- Install cuDNN
  - manula [https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-deb)
  - archive [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)
  - .deb말고(힘듬) .tar를 받자(쉬움)

- Check properly installed
  - files > 다른 위치 > 컴퓨터 > usr > local > cuda > include > cudnn*


- Install (.deb 오히려 좀 더 귀찮음)
  - CUDA toolkit에 포함되어 있다고 한다. ~~안되는데~~
  - Go to official site [https://developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download)
  - Download proper version
    - Local Installers for Windows and Linux, Ubuntu(x86_64, armsbsa)
    - Local Installer for Ubuntu20.04 x86_64 (Deb)
    - CUDA 11.8
  - GPG key
    ```bash
    sudo cp /var/cudnn-local-repo-ubuntu2004-8.9.5.29/cudnn-local-C1AC07B2-keyring.gpg /usr/share/keyrings/
    ```
  - depackage
    ```bash
    sudo dpkg -i cudnn-local-repo-ubuntu2004-8.9.5.29_1.0-1_amd64.deb
    ```
  - apt update
    ```bash
    sudo apt-get update
    ```
  - install
    ```bash
    sudo apt-get install cudnn-local-repo-ubuntu2004-8.9.5.29
    ```
  - add environment variable
    ```bash
    sudo find /usr -name cuda
    # outputs
    # /usr/include/thrust/system/cuda
    # /usr/local/cuda-11.8/targets/x86_64-linux/include/thrust/system/cuda
    # /usr/local/cuda-11.8/targets/x86_64-linux/include/cuda
    # /usr/local/cuda
    # /usr/share/doc/cuda
    # /usr/share/doc/libthrust-dev/examples/cuda
    export PATH=/usr/local/cuda-11.8/bin:/usr/local/cuda-11.8/NsightCompute-2023.1:$PATH
    ```

$disto = ubuntu2004
x.x.x = 8.9.5.29
X.Y = 11.8


- Installed location
  ```bash
  dpkg -L cudnn-local-repo-ubuntu2004-8.9.5.29
  # /.
  # /etc
  # /etc/apt
  # /etc/apt/sources.list.d
  # /etc/apt/sources.list.d/cudnn-local-ubuntu2004-8.9.5.29.list
  # /usr
  # /usr/share
  # /usr/share/doc
  # /usr/share/doc/cudnn-local-repo-ubuntu2004-8.9.5.29
  # /usr/share/doc/cudnn-local-repo-ubuntu2004-8.9.5.29/changelog.Debian.gz
  # /var
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/C1AC07B2.pub
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/InRelease
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/Local.md5
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/Local.md5.gpg
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/Packages
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/Packages.gz
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/Release
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/Release.gpg
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/cudnn-local-C1AC07B2-keyring.gpg
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/libcudnn8-dev_8.9.5.29-1+cuda11.8_amd64.deb
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/libcudnn8-samples_8.9.5.29-1+cuda11.8_amd64.deb
  # /var/cudnn-local-repo-ubuntu2004-8.9.5.29/libcudnn8_8.9.5.29-1+cuda11.8_amd64.deb
  ```

- installed location is 컴퓨터 > var > cudnn-local~~~

- add cudnn-local repository on system (using package manager)
  ```bash
  sudo cp /var/cudnn-local-repo-ubuntu2004-8.9.5.29/cudnn-local-C1AC07B2-keyring.gpg /usr/share/keyrings/
  echo "deb [signed-by=/usr/share/keyrings/cudnn-local-C1AC07B2-keyring.gpg] file:/var/cudnn-local-repo-ubuntu2004-8.9.5.29/ /" | sudo tee /etc/apt/sources.list.d/cudnn-local-ubuntu2004-8.9.5.29.list
  sudo apt-get update
  sudo apt-get install libcudnn8
  ```

- TensorRT install and merge to Tensorflow
  - download [https://developer.nvidia.com/nvidia-tensorrt-8x-download](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
  - TensorRT 8.6 GA for Ubuntu 20.04 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7 and 11.8 DEB local repo Package
- dpkg
- 저장소 키 업데이트 sudo cp /var/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-11.8/nv-tensorrt-local-D7BB1B18-keyring.gpg /usr/share/keyrings/



- Tar version







Version

https://robot9710.tistory.com/29


https://jjuke-brain.tistory.com/entry/GPU-%EC%84%9C%EB%B2%84-%EC%82%AC%EC%9A%A9%EB%B2%95-CUDA-PyTorch-%EB%B2%84%EC%A0%84-%EB%A7%9E%EC%B6%94%EA%B8%B0-%EC%B4%9D%EC%A0%95%EB%A6%AC


https://www.tensorflow.org/install/pip?hl=ko

https://jjuke-brain.tistory.com/entry/GPU-%EC%84%9C%EB%B2%84-%EC%82%AC%EC%9A%A9%EB%B2%95-CUDA-PyTorch-%EB%B2%84%EC%A0%84-%EB%A7%9E%EC%B6%94%EA%B8%B0-%EC%B4%9D%EC%A0%95%EB%A6%AC 
