---
layout: single
title: "Generate github blog"
categories: blog
tags: [Github, Blog, Installation]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

# Generation
## Generate github blog
- Refer to: 
    - https://www.youtube.com/watch?v=ACzFIAOsfpM
- In short: 
    - Go to template link https://github.com/topics/jekyll-theme
    - Folk
    - Change repository name to (my ID).github.io
  
## Enter github blog
- https://koyumi0601.github.io/

## Renew page
- Control + R

## Delete github repository
- Refer to: 
    - https://coding-factory.tistory.com/246
- In short: 
    - Go to target repository > Settings > Danger zone > Delete this repository

# Management
## Webserver Local Hosting
- for fast debugging
- install ruby
```bash
sudo apt-get install ruby-dev
```
- clone target repository
- Go to project folder and opne in terminal
- build website
```bash
sudo bundle install
sudo bundle exec jekyll serve
```
- Go to http://127.0.0.1:4000
- Stop local hosting
```bash
Control + C
```
- Renew: stop and rerun jekyll serve

## Code highlight Failure (Python)
아래의 방법으로 시도 중이며 아직 완료하지 않음.
- template doesn't support python code highlight
- Install ruby and plug-in(jekyll-pygments)
### steps:
- Install ruby 
```bash
sudo apt install ruby-bundler 
# sudo snap install ruby --classic
# sudo apt-get install ruby-dev
sudo bundle install
sudo gem update bundler # dependency
```
- Check gem(jekyll-pygments) is installed already
```bash
gem list
```

- Go to project file and generate Gemfile. File this exists, add lines like below
```
source "https://rubygems.org"
gem "jekyll"
gem "jekyll-pygments" # jekyll-pygments 플러그인 추가
```
- Go to project file and install plug-in
```bash
bundle install
```
