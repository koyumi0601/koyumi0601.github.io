---
layout: single
title: "How to add search on navigation"
categories: blog
tags: [Github, Blog, Template]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---



# Work Instruction

- Add file _pages/search.md

![2023-08-03_14-29-search]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_14-29-search.png)

```markdown
---
title: Search
layout: search
permalink: /search/
---
```

- Edit navigation.yml

![2023-08-03_14-29-search-nav]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_14-29-search-nav.png)

```yml
main:
  - title: "Category"
    url: /categories/
  - title: "Tag"
    url: /tags/
  - title: "Search"
    url: /search/
```

- in post, write below line

![2023-08-03_14-29-search-post]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_14-29-search-post.png)

```markdown
search: true
```

- Result

![2023-08-04_17-57-search1]({{site.url}}/images/2023-08-03-Github-Blog-Posting-Search/2023-08-04_17-57-search1.png)




# Reference

- 테디노트 영상 [https://www.youtube.com/watch?v=AONVKTeeaWY&t=2s](https://www.youtube.com/watch?v=AONVKTeeaWY&t=2s)