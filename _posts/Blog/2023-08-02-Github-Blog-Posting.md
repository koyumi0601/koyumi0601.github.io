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


# Purpose

*This post is to record tips to post on Github blog using template; minimal mistake*



# Terminology

***Minimal Mistake** - a popular template among Jekyll themes, widely used for creating blogs and portfolios*

***Jekyll** - a static site generator developed in Ruby, used to create static websites using simple Markdown syntax and HTML/CSS*



# Body

## Add Category and Tag

*This is how to edit navigation bar like below*



![2023-08-03_13-34-5]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-34-5.png)

- Go to (local project path)/_config.yml
- Undo comment

![2023-08-03_13-12-Add-Category]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-12-Add-Category.png)

- Make file (local project path)/_pages/category-archive.md

![2023-08-03_13-23-Add-category-2]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-23-Add-category-2.png)

```markdown
---
title: "Category"
layout: categories
permalink: /categories/
author_profile: true
sidebar_main: true
---
```

- Make file (local project path)/_pages/tag-archive.md

![2023-08-03_13-23-Add-category-3]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-23-Add-category-3.png)

```markdown
---
title: "Tag"
layout: tags
permalink: /tags/
author_profile: true
sidebar_main: true
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
  - title: "Search"
    url: /search/
```

![2023-08-03_13-23-Add-category-4]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-23-Add-category-4.png)

- Go to (local project path)/_posts/(sample category name)/(sample post).md

![2023-08-03_13-34-6]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-34-6.png)

- Add below lines

```markdown
categories: Blog
tags: [Github, Blog, Template]
```

- Result

![2023-08-03_13-46-7-result]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-46-7-result.png)



## Add TOC (Table of Contents)

*This is how to add table of contents like below*

![2023-08-03_13-52-toc-result]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-52-toc-result.png)

- In post, write below line

![2023-08-03_13-49-toc]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-49-toc.png)

```markdown
toc: true
```



## Disable/Enable Author Profile

*This is how to disable author profile on the post*



- in post, write below line

![2023-08-03_13-55-authorprofile]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-55-authorprofile.png)



```markdown
author_profile: false
```

![2023-08-03_13-55-authorprofile-disable]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-55-authorprofile-disable.png)



```markdown
author_profile: true
```

![2023-08-03_13-55-authorprofile-enable]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-55-authorprofile-enable.png)



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



## Add Search

- add file _pages/search.md

![2023-08-03_14-29-search]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_14-29-search.png)

```markdown
---
title: Search
layout: search
permalink: /search/
---
```
- edit navigation.yml

![2023-08-03_14-29-search-nav]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_14-29-search-nav.png)

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

![2023-08-03_14-29-search-post]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_14-29-search-post.png)

```markdown
search: true
```

- Result

![2023-08-03_14-29-search-nav-ui]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_14-29-search-nav-ui.png)

![2023-08-03_14-29-search-result-in-searchpage]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_14-29-search-result-in-searchpage.png)





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



## Change Font Size

- Go to (local project folder)/_sass_minimal-mistakes/_reset.scss

- Change font-size: 16px; to 10px; and others

  ![2023-08-03_13-00-font-size-change]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_13-00-font-size-change.png)



## Add Notice

- Refer to: https://www.youtube.com/watch?v=q0P3TSoVNDM&t=184s
- In short:
<div class="notice">
    <ul>
        <li> Some notice </li>
    </ul>
</div>
{: .notice--danger}



## Add Youtube Clip

- Actually, remove &t=xxs in id q0P3TSoVNDM&t=184s
{% include video id="q0P3TSoVNDM" provider="youtube" %}



## Add Image

- Download typora [https://typora.io/#linux](https://typora.io/#linux)
- File > Preferences > Image > 'copy image to custom folder' > ../../images/$(filename)
  - Or ../images (it depends on your _post file structure)
- Drag and drop image to post
- Edit absolute path to relative path like ![2023-08-03_15-13-image]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-03_15-13-image.png)

# Reference

{% include video id="3UOh0rKlxjg" provider="youtube" %}

Author profile, Sidebar, Search

{% include video id="AONVKTeeaWY" provider="youtube" %}

Sidebar (by categories)

Sunghwan's blog [https://sunghwan7330.github.io/blog/blog_sidebar/](https://sunghwan7330.github.io/blog/blog_sidebar/)

