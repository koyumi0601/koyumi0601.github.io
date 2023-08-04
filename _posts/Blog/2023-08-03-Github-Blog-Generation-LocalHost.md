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


# Reference
- Some blogs
- 테디노트 영상 [https://www.youtube.com/watch?v=ACzFIAOsfpM](https://www.youtube.com/watch?v=ACzFIAOsfpM])