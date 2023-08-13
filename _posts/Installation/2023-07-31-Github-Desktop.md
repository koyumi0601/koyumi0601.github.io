---
layout: single
title: "How to install Github Desktop in Ubuntu"
categories: setup
tags: [Github Desktop, Blog, Ubuntu, Installation]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

*Simplified Git management tool*

# Work Instruction

- Install git

```bash
sudo apt install git
```

- Install Github desktop in Ubuntu

```bash
sudo apt-get update
sudo apt-get install snapd
wget https://github.com/shiftkey/desktop/releases/download/release-2.0.4-linux1/GitHubDesktop-linux-2.0.4-linux1.snap
sudo snap install GitHubDesktop-linux-2.0.4-linux1.snap --classic --dangerous
```
- Go to Activities and Search github desktop

- Download to local using github desktop
  - Download github desktop and vs code
  - Open github desktop and login

# Trial and Error  
## Authentication Failure

- Go to github [https://github.com/](https://github.com/)
- Go to right upper corner and click settings
- Go to left lower corner  and click developer settings
- Go to personal access toekns
- Go to Tokens (classic)
- Click Generate new token or regenerate if you already have it
- Copy and paste token to Password
- register user name and password to local pc git
```bash
git config --global user.name 'your_id@your_email.com'
git config --global user.password ‘copied_token’
```
*In short, password = token*


# Reference 
- [https://davelogs.tistory.com/55](https://davelogs.tistory.com/55)
- [https://wotres.tistory.com/entry/Github-%EC%97%90%EB%9F%AC-%ED%95%B4%EA%B2%B0%EB%B2%95-Authentication-failed-for-use-a-personal-access-token-instead](https://wotres.tistory.com/entry/Github-%EC%97%90%EB%9F%AC-%ED%95%B4%EA%B2%B0%EB%B2%95-Authentication-failed-for-use-a-personal-access-token-instead)