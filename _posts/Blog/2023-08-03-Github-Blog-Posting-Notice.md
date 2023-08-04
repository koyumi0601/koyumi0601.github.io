---
layout: single
title: "How to add notice(emphasize paragraph) on a post"
categories: blog
tags: [Github, Blog, Template]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

# Work Instruction

- Wrap paragraph with class notice
- add at the end of the paragraph

```html
<div class="notice">
    <li> Some notice 1 </li>
    <li> Some notice 2 </li>
</div>{: .notice--danger}
```






- Result

<div class="notice">
    <li> Some notice 1 </li>
    <li> Some notice 2 </li>
</div>{: .notice--danger}






# Reference

- 테디노트 영상 [https://www.youtube.com/watch?v=q0P3TSoVNDM&t=184s](https://www.youtube.com/watch?v=q0P3TSoVNDM&t=184s)
- Minimal mistake Utility Classes [https://mmistakes.github.io/minimal-mistakes/docs/utility-classes/#notices](https://mmistakes.github.io/minimal-mistakes/docs/utility-classes/#notices)