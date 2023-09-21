---
layout: single
title: "How to Share folder between window and ubuntu, samba"
categories: setup
tags: [Ubuntu, Installation]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*SAMBA*



# Install SAMBA

- Window PC

  - 기본으로 깔려있음. 

  - window + R, service에서 service라는 이름으로 실행 중인지 확인.

- Ubuntu

  - Install Samba

    ```bash
    sudo apt update
    sudo apt install smbclient
    ```

  - Access shareforlder

    ```bash
    smbclient //Windows_PC_IP_Address/공유_이름 -U 사용자_이름
    ```

    
