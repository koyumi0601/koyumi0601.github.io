---
layout: single
title: "How to add category and tag on navigation"
categories: blog
tags: [Github, Blog, Template]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---



# Work Instruction

- Go to (local project path)/_config.yml
- Undo comment

```markdown
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



- Make file (local project path)/_pages/category-archive.md
  - sidebar option: main, docs, (not defined)


```markdown
---
title: "Category"
layout: categories
permalink: /categories/
author_profile: false
sidebar_main: true
sidebar:
    nav: "docs"
---
```



- Make file (local project path)/_pages/tag-archive.md

```markdown
---
title: "Tag"
layout: tags
permalink: /tags/
author_profile: false
sidebar_main: true
sidebar:
    nav: "docs"
---
```



- Go to (local project path)/_data/navigation.yml
- Add below lines

```markdown
main:
  - title: "Category"
    url: /categories/
  - title: "Tag"
    url: /tags/
```

- Go to (local project path)/_posts/(sample category name)/(sample post).md

- Add below lines

```markdown
categories: Blog
tags: [Github, Blog, Template]
```

- for example, 

```markdown
---
layout: single
title: "How to add category and tag on navigation"
categories: blog
tags: [Github, Blog, Template]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---
```

- Result

![2023-08-04_20-52-category]({{site.url}}/images/2023-08-03-Github-Blog-Posting-Category-Tag/2023-08-04_20-52-category.png)


# Reference

- 테디노트 영상 [https://www.youtube.com/watch?v=3UOh0rKlxjg](https://www.youtube.com/watch?v=3UOh0rKlxjg)
