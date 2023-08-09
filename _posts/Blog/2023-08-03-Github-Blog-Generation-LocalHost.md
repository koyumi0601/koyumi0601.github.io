---
layout: single
title: "How to review updated post (webserver local hosting)"
categories: blog
tags: [Github, Blog, Installation]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

# Work Instruction

- for fast debugging, *bundle exec jekyll serve* on http://127.0.0.1:4000



### install ruby and bundler

#### Ubuntu

```bash
sudo apt-get install ruby-dev
sudo bundle install
```
- Clone target repository
- Go to project folder and open in terminal
- build website

```bash
sudo bundle exec jekyll serve
```

- Go to http://127.0.0.1:4000
- Stop local hosting

```bash
Control + C
```

- Renew: stop and rerun jekyll serve

#### Windows

- Ruby(dev kit) :download [https://rubyinstaller.org/](https://rubyinstaller.org/)
- (Install Msys during Ruby installation)
- Go to cmd and install bundler

```bash
gem install bundler 
```

- Install jekyll

```bash
gem install jekyll
```

- Go to project foler and install bundle

```bash
bundle install
```

- run jekyll

```bash
bundle exec jekyll serve
```

- Go to http://127.0.0.1:4000/ with any explorer






# Reference
- Some blogs
- 테디노트 영상 [https://www.youtube.com/watch?v=ACzFIAOsfpM](https://www.youtube.com/watch?v=ACzFIAOsfpM])

- Windows에서 Github와 Jekyll 개발 환경 설치하기 [https://wormwlrm.github.io/2018/07/13/How-to-set-Github-and-Jekyll-environment-on-Windows.html](https://wormwlrm.github.io/2018/07/13/How-to-set-Github-and-Jekyll-environment-on-Windows.html)