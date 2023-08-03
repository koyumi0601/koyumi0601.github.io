---
layout: single
title: "Install Github Desktop"
categories: setup
tags: [Github Desktop, Blog, Ubuntu, Installation]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

# Install Github desktop in Ubuntu
- Refer to: 
    - https://davelogs.tistory.com/55 
- In short:
```bash
sudo apt-get update
sudo apt-get install snapd
wget https://github.com/shiftkey/desktop/releases/download/release-2.0.4-linux1/GitHubDesktop-linux-2.0.4-linux1.snap
sudo snap install GitHubDesktop-linux-2.0.4-linux1.snap --classic --dangerous
```
- Go to Activities and Search github desktop

# Download to local using github desktop
- Download github desktop and vs code
- Open github desktop and login
  
## Authentication Failure
- Refer to: 
    - https://wotres.tistory.com/entry/Github-%EC%97%90%EB%9F%AC-%ED%95%B4%EA%B2%B0%EB%B2%95-Authentication-failed-for-use-a-personal-access-token-instead
- In short: 
    <div class="notice">
        <ul>
            <li> Password = token </li>
        </ul>
    </div>
    {: .notice--danger}
