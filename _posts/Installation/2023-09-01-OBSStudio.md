---
layout: single
title: "How to install OBS Studio"
categories: setup
tags: [Ubuntu, Installation]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*Screen and sound record in Ubuntu*

- install OBS Studio

```bash
sudo apt update
sudo apt install obs-studio
```

- Use
    - 애플리케이션 메뉴에서 OBS Studio 실행
    - 하단의 + 버튼 클릭, 새 씬(Scene) 추가
    - 'Sources' 박스에서 + 버튼 클릭, 원하는 소스(Display Capture, Window Capture 등) 추가
    - 'Audio Mixer'에서 오디오 설정
    - 상단 메뉴의 Settings -> Output에서 녹화 설정
    - 하단의 Start Recording 버튼 클릭, 녹화 시작
    - Stop Recording 버튼 클릭, 녹화 중지
    - 필요시 Settings -> Stream에서 스트리밍 서비스 연동 후 Start Streaming 버튼 클릭, 스트리밍 시작
