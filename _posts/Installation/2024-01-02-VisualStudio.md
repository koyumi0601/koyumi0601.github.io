---
layout: single
title: "How to install VS Code in Ubuntu"
categories: setup
tags: [Visual Studio Code, VS Code, Ubuntu, Installation]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---
*Code editor made by Microsoft*

# Ubuntu 
- Work Instruction

- Update latest ubuntu packages
```bash
sudo apt update
```

- Install required packages
  - software-properties-common: include tools to save software repository
  - apt-transport-https: download packages through https
  - wget: download files through web
```bash
sudo apt install software-properties-common apt-transport-https wget
```

- Download GPG key from Microsoft and add it on system. Now, Microsoft package repository is reliable.
```bash
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
```
- Add Visual Studio Code repository to ubuntu software repository. Now, Visual Studio Code can be installed through ubuntu package manager
```bash
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
```

- Install Visual Studio Code and Execute it
```bash
sudo apt install code
```

- Go to activities > Search VS code 

- Run as admin
    - library separated with user. need to install
```bash
sudo code --no-sandbox --user-data-dir=~/Documents/vscode
```

- Update vs code as site suggrest
    - go to site
    - download .deb
    - go to download folder and open terminal
    - sudo dpkg -i filename.deb


# Cpp build and execute using VS code in Ubuntu

- Install g++
```bash
sudo apt-get install g++
```

- Build
```bash
g++ -o output_filename source_filename.cpp
```

- Another way

```bash
g++ main.cpp -o main_c
./main_c
```


- Build automation (optional)
  - generate .vscode folder in project folder
  - generate taks.json file in .vscode folder

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build",
            "type": "shell",
            "command": "g++",
            "args": [
                "-o",
                "output_filename",
                "source_filename.cpp"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```
- Ctrl + shift + B
- execute

```bash
./output_filename
```

## Reference
- [https://jjeongil.tistory.com/1951](https://jjeongil.tistory.com/1951)


## Graphic driver install in VS code

- environment setting difference with terminal
- in VS code terminal, 
```bash
apt install nvidia-cuda-toolkit
```
- system environment path edit in ~/.bashrc 
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```





# window

## python
```bash
python --version # version check
(Get-Command python).Source # 설치경로, PowerShell
which python # unix 계열 운영체제
where python # cmd.. 근데 안나옴
```
```
D:\Program Files\Python311\python.exe # 출력
```
- 파이썬 다른 것으로 변경하고 싶다면
- .vscode/settings.json 생성
- 프로젝트별로 환경 설정 가능
```
{
    "python.pythonPath": "/원하는/파이썬/경로/python.exe"
}
```
## C++ 컴파일러 
- g++
- msys2를 설치한 후 g++를 설치할 수 있다.
```bash
g++ --version # 설치 확인
```

## .cu 컴파일러
- CUDA toolkit 설치해야 한다. [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
```bash
nvcc --version
```
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Nov__3_17:51:05_Pacific_Daylight_Time_2023
Cuda compilation tools, release 12.3, V12.3.103
Build cuda_12.3.r12.3/compiler.33492891_0
```


### 오류...

- nvcc error   : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)

### Graphic driver check

- **NVIDIA GPU 그래픽카드** : 

  - 장치관리자 > NVIDIA GeForce RTX 3060

- **NVIDIA GPU 드라이버** : 

  - https://www.nvidia.co.kr/download/driverResults.aspx/216965/kr 

  - 546.33-desktop-win10-win11-64bit-international-dch-whql.exe

    - ## GeForce Game Ready 드라이버

       

      | 버전:      | 546.33 **WHQL**               |
      | ---------- | ----------------------------- |
      | 배포 날짜: | 2023.12.12                    |
      | 운영 체제: | Windows 10 64-bit, Windows 11 |
      | 언어:      | Korean                        |
      | 파일 크기: | 669.39 MB                     |

  - D:\NVIDIA\DisplayDriver\546.33\Win11_Win10-DCH_64\International

  - 시작 > NVIDIA Control Panel > Help > System Information > 자세한 정보를 얻을 수 있다
  - 장치관리자 > Display adapters > NVIDIA GeForce RTX 3060 > 더블 클릭 > 31.0.15.4633

### Graphic driver and toolkit compatibility
- [https://docs.nvidia.com/deploy/cuda-compatibility/](https://docs.nvidia.com/deploy/cuda-compatibility/)
- 그래픽 드라이버와 이에 따라 추천되는 cuda version 한번에 확인하기, nvidia-smi
```
PS D:\GitHub_Project\koyumi0601.github.io> nvidia-smi
Tue Jan  2 16:00:11 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 546.33                 Driver Version: 546.33       CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3060      WDDM  | 00000000:2B:00.0  On |                  N/A |
|  0%   35C    P8              22W / 170W |    426MiB / 12288MiB |      5%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A       996    C+G   ...2txyewy\StartMenuExperienceHost.exe    N/A      |
|    0   N/A  N/A      1780    C+G   ...ekyb3d8bbwe\PhoneExperienceHost.exe    N/A      |
|    0   N/A  N/A      2812    C+G   ...oogle\Chrome\Application\chrome.exe    N/A      |
|    0   N/A  N/A      4312    C+G   ...Programs\Microsoft VS Code\Code.exe    N/A      |
|    0   N/A  N/A      5204    C+G   C:\Windows\explorer.exe                   N/A      |
|    0   N/A  N/A      7644    C+G   ...siveControlPanel\SystemSettings.exe    N/A      |
|    0   N/A  N/A      8536    C+G   ....Search_cw5n1h2txyewy\SearchApp.exe    N/A      |
|    0   N/A  N/A      9832    C+G   ...CBS_cw5n1h2txyewy\TextInputHost.exe    N/A      |
|    0   N/A  N/A     11560    C+G   ...64__8wekyb3d8bbwe\CalculatorApp.exe    N/A      |
+---------------------------------------------------------------------------------------+
```
- 실제 설치된 cuda version 확인
```bash
nvcc --version
```

- 참고: [https://tkayyoo.tistory.com/17](https://tkayyoo.tistory.com/17)


## 빌드 및 실행
- .vscode/tasks.json
```
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "C++ Compile",
            "type": "shell",
            "command": "g++",
            "args": ["-o", "cpp_output.exe", "main.cpp"],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        },
        {
            "label": "CUDA Compile",
            "type": "shell",
            "command": "nvcc",
            "args": ["-o", "cuda_output.exe", "main.cu"],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        },
        {
            "label": "Run Python Script",
            "type": "shell",
            "command": "python",
            "args": ["script.py"],
            "group": {
                "kind": "test",
                "isDefault": true
            },
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        }
    ]
}
```
- Ctrl+Shift+B


## msys2
- python
```
export PATH="/원하는/파이썬/경로:$PATH"
export PATH="/d/Program Files/Python311:$PATH"
# MSYS2에서는 Windows 드라이브를 /c/, /d/, /e/ 등과 같은 심볼릭 링크로 표현하고, 경로 구분자로 슬래시(/)를 사용
# which python 으로 확인
```
- g++ 컴파일러 설치
```
pacman -Syu # 패키지관리자 업데이트
pacman -Su # 시스템에 설치된 패키지를 최신으로 업데이트
pacman -S gcc
g++ --version # 설치 버전 확인
```