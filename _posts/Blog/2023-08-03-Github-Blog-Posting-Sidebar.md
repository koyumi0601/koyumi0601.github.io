---
layout: single
title: "How to add sidebar using categories"
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
docs:
  - title: Electrical Engineering
    children:
      - title: "Machine Learning"
        url: /categories/machinelearning
      - title: "Image Signal Processing"
        url: /categories/imagesignalprocessing
      - title: "Digital Signal Processing"
        url: /categories/digitalsignalprocessing
  - title: Natural Sciences
    children:
      - title: "Physics"
        url: /categories/physics
      - title: "Linear Algebra"
        url: /categories/linearalgebra
  - title: Computer Science
    children:
      - title: "Python"
        url: /categories/python
      - title: "CUDA"
        url: /categories/cuda    
  - title: Others
    children:
      - title: "Setup"
        url: /categories/setup
      - title: "Blog"
        url: /categories/blog
      - title: "Book"
        url: /categories/book
      - title: "House Chores"
        url: /categories/housechores
```

- Add files in _pages/categories/machinelearning.md, imagesignalprocessing.md and so on
- in case of blog.md, contents are like below

![2023-08-04_21-33-doc]({{site.url}}/images/2023-08-03-Github-Blog-Posting-Sidebar/2023-08-04_21-33-doc.png)

- Go to _config.yml and add sidebar: nav: "docs" like below

```yml
defaults:
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      share: true
      related: true
      sidebar:
        nav: "docs"
```

- in post, add lines like below

```markdown
---
layout: single
title: "How to add sidebar using categories"
categories: blog
tags: [Github, Blog, Template]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---
```





### Add sidebar on other page

- Home: index.html

![2023-08-03_17-49-home-sidebar]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_17-49-home-sidebar.png)

- each page: _pages/*.md

![2023-08-04_10-59-sidebar-search]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-04_10-59-sidebar-search.png)








# Reference

- Sunghwan's blog [https://sunghwan7330.github.io/blog/blog_sidebar/](https://sunghwan7330.github.io/blog/blog_sidebar/)

- 공부하는 식빵맘 blog [https://ansohxxn.github.io/blog/category/](https://ansohxxn.github.io/blog/category/)


