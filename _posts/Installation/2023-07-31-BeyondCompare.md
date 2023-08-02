---
layout: single
title: "How to install Beyond Compare (Linux)"
categories: Setup
tags: [Beyond Compare, Ubuntu, Installation]
toc: true
author_profile: false
---
# Linux license
## Terminal Installation
- Refer to:
    - https://www.scootersoftware.com/kb/linux_install
- In short:
```bash
wget https://www.scootersoftware.com/bcompare-4.4.6.27483_amd64.deb
sudo apt update
sudo apt install ./bcompare-4.4.6.27483_amd64.deb
```
```bash
cd /usr/bin # go to installed path
bcompare # execute
```
- Move to unpacked folder and execute beyond compare
<s>
## Add to dropdown menu with right click
### Install alacarte
```bash
sudo apt-get update
sudo apt-get install alacarte
```
- Go to Activities
- Search Main Menu
- Select Graphics
- New item
</s>

## Uninstall
```bash
sudo apt remove bcompare
```

# Window license
- 실행 시 Error 다수 발생하여 적절해 보이지 않음
## Download window version
- https://www.scootersoftware.com/download
## Install
```bash
sudo apt-get update
sudo apt-get install wine
wine BCompare-4.4.6.27483.exe
```
## Execute
```bash
wine "/home/사용자이름/.wine/drive_c/Program Files/Beyond Compare 4/BCompare.exe"
```
- 사용자이름: 현재 사용자의 홈 디렉토리명
- drive_c/Program Files/Beyond Compare 4/BCompare.exe: 설치된 경로
```bash
wine "/home/koyumi/.wine/dosdevices/c:/Program Files/Beyond Compare 4/BCompare.exe"
```