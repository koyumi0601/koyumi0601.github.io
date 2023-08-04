---
layout: single
title: "How to add google analytics"
categories: blog
tags: [Blog, Markdown]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---



# Work Instruction

- Go to google analytics [https://analytics.google.com/analytics/web](https://analytics.google.com/analytics/web)
- Get google analytic Id
- Go to _config.yml
- Edit provider, tracking_id, anonymize_ip like below

```yml
# Analytics
analytics:
  provider               : "google-gtag" # false (default), "google", "google-universal", "google-gtag", "custom"
  google:
    tracking_id          : "your id generated from google analytics"
    anonymize_ip         : false # true, false (default)
```

- Go to google analytics web and check report




# Reference

- 테디노트 영상 [EP05. 댓글 & 구글 애널리틱스 추가하기 https://www.youtube.com/watch?v=anXaW9xhgcU](https://www.youtube.com/watch?v=anXaW9xhgcU)


- Minimal mistake configuration [https://mmistakes.github.io/minimal-mistakes/docs/configuration/#skin](https://mmistakes.github.io/minimal-mistakes/docs/configuration/#skin)