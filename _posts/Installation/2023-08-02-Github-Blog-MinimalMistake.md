---
layout: single
title: "Edit Template - Minimal Mistake"
categories: Blog
tags: [Github, Blog, Template]
toc: true
author_profile: false
---
# Template - Minimal Mistake
## Add category
- Refer to:
    - https://www.youtube.com/watch?v=3UOh0rKlxjg
- In short:

1. _config.yml
- 주석 취소
```yml
jekyll-archives:
  enabled:
    - categories
    - tags
  layouts:
    category: archive-taxonomy
    tag: archive-taxonomy
  permalinks:
    category: /categories/:name/
    tag: /tags/:name/
```

2. _data/navigation.yml
- add navigation
```yml
main:
  - title: "Category"
    url: /categories/
  - title: "Tag"
    url: /tags/
```
3. _pages/category-archive.md
- add folder _pages
- add category-archive.md
```markdown
---
title: "Category"
layout: categories
permalink: /categories/
author_profile: true
sidebar_main: true
---
```
4. _pages/tag-archive.md
```markdwon
---
title: "Tag"
layout: tags
permalink: /tags/
author_profile: true
sidebar_main: true
---
```
## Add TOC (Table of Contents)
- in post, write below line
```markdown
toc: true
```
## 404 Page Error
- generate file _pages/404.md refer to test/_pages/404.md

## Disable/Enable Author Profile
- in post, write below line
```markdown
author_profile: false
```