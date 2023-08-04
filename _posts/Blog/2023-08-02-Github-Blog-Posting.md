---
layout: single
title: "How to post on Github blog using template"
categories: blog
tags: [Github, Blog, Template]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---




## Add Sidebar

### Sidebar using Category and Tag

*Not preferred*

- Add below lines in _data/navigation.yml

![2023-08-03_14-13-sidebar]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_14-13-sidebar.png)

```yaml
docs:
  - title: "대목차1"
  - children:
    - title: "Category"
      url: /categories/
    - title: "Tag"
      url: /tags/
  - title: "대목차2"
  - children:
    - title: "Category"
      url: /categories/
    - title: "Tag"
      url: /tags/
```
- in post, write below line

![2023-08-03_14-13-sidebar-post]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_14-13-sidebar-post.png)

```markdown
sidebar:
    nav: "docs"
```

- Result

![2023-08-03_14-13-sidebar-result]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_14-13-sidebar-result.png)





### Sidebar using categories

https://sunghwan7330.github.io/blog/blog_sidebar/



### Add sidebar on other page

- Home: index.html

![2023-08-03_17-49-home-sidebar]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_17-49-home-sidebar.png)

- each page: _pages/*.md

![2023-08-04_10-59-sidebar-search]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-04_10-59-sidebar-search.png)








# Reference

{% include video id="3UOh0rKlxjg" provider="youtube" %}

Author profile, Sidebar, Search

{% include video id="AONVKTeeaWY" provider="youtube" %}

Sidebar (by categories)

Sunghwan's blog [https://sunghwan7330.github.io/blog/blog_sidebar/](https://sunghwan7330.github.io/blog/blog_sidebar/)

공부하는 식빵맘 blog [https://ansohxxn.github.io/blog/category/](https://ansohxxn.github.io/blog/category/)

Equation
[https://an-seunghwan.github.io/github.io/mathjax-error/](https://an-seunghwan.github.io/github.io/mathjax-error/)

[https://ashki23.github.io/markdown-latex.html](https://ashki23.github.io/markdown-latex.html)