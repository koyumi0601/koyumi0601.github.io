---
layout: single
title: "How to enable/disable author profile"
categories: blog
tags: [Github, Blog, Template]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---



# Work Instruction

- Post: In post, add lines 'author_profile': false
- Home: index.html, edit lines 'author_profile': false
- Pages (Category, Tag, Search, Other categories): _pages/*.md, edit lines 'author_profile': false

```markdown
---
title: "Blog"
layout: archive
permalink: categories/blog
author_profile: false
types: posts
sidebar:
  nav: "docs"
---
```

- Result

- author_profile: true

![2023-08-04_18-24-authorprofile-true]({{site.url}}/images/2023-08-03-Github-Blog-Posting-AuthorProfile/2023-08-04_18-24-authorprofile-true.png)

- author_profile: false

![2023-08-04_18-24-authorprofile-false]({{site.url}}/images/2023-08-03-Github-Blog-Posting-AuthorProfile/2023-08-04_18-24-authorprofile-false.png)






# Reference

