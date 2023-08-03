---
layout: single
title: "Edit Template - Minimal Mistake"
categories: Blog
tags: [Github, Blog, Template]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
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
## Add 404 Page Error
- generate file _pages/404.md refer to test/_pages/404.md

## Disable/Enable Author Profile
- in post, write below line
```markdown
author_profile: false
```

## Add Sidebar
- add below lines in _data/navigation.yml
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
```markdown
sidebar:
    nav: "docs"
```

## Add Search
- add file _pages/search.md
```markdown
---
title: Search
layout: search
permalink: /search/
---
```
- edit navigation.yml
```markdown
main:
  - title: "Category"
    url: /categories/
  - title: "Tag"
    url: /tags/
  - title: "Search"
    url: /search/
```
- in post, write below line
```markdown
search: true # default true
```

## Change Font
- Go to https://fonts.google.com/
- Select Korean
- Select Noto Sans Korean (example)
- View Selected family
- See 'Use on the web'
- Select @import
- Copy @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300&display=swap');
- Go to _sass/minimal-mistakes.scss
- Add below lines
```scss
/* Google fonts */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300&display=swap');
```
- Go to google font and check font family name in 'CSS rules to specify families'
- Go to _sass/minimal-mistakes/_variables.scss
- Edit like below
```scss
/* system typefaces */
$serif: Georgia, Times, serif !default;
$sans-serif: -apple-system, BlinkMacSystemFont, "Noto Sans KR", "Roboto", "Segoe UI",
  "Helvetica Neue", "Lucida Grande", Arial, sans-serif !default;
$monospace: Monaco, Consolas, "Lucida Console", monospace !default;

```

## Add Notice
- Refer to: https://www.youtube.com/watch?v=q0P3TSoVNDM&t=184s
- In short:
<div class="notice">
    <ul>
        <li> Some notice </li>
    </ul>
</div>
{: .notice--danger}

## Add Button
[This is a button](https://google.com){: .btn .btn--danger}

## Add Youtube
- Actually, remove &t=xxs in id q0P3TSoVNDM&t=184s
{% include video id="q0P3TSoVNDM" provider="youtube" %}



## Insert Image

- Download typora https://typora.io/#linux
- File > Preferences > Image > 'copy image to custom folder' > ../images/$(filename)
- Drag and drop image to post
