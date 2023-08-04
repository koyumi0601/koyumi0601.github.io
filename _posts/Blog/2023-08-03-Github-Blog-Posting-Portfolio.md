---
layout: single
title: "How to add portfolio(single page in main navigation)"
categories: blog
tags: [Github, Blog, Template]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---



# Work Instruction

- Go to _data/navigation.yml
- Add below lines

```yml
main:
  - title: "Portfolio"
    url: /portfolio/
```
- Add page in _pages/portfolio.md

```yml
---
title: Portfolio
layout: single
permalink: /portfolio/
author_profile: true
# sidebar:
#     nav: "docs"
---
```

- Fill in anything like post below area in portfolio.md


# Reference

- None
