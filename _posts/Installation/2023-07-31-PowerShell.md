---
layout: single
title: "How to install powershell"
categories: Setup
tags: [Power Shell, Ubuntu, Installation]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---
# add PowerShell repository
```bash
wget -q https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
```

# Update repository:
```bash
sudo apt-get update
```

# Install PowerShell:
```bash
sudo apt-get install -y powershell
```

# Run
```bash
pwsh
```
