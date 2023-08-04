---
layout: single
title: "How to change font on a post"
categories: blog
tags: [Github, Blog, Template]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---



# Work Instruction

## Change Font

- Go to Google Fonts [https://fonts.google.com/](https://fonts.google.com/)
- Select Korean and Noto Sans Korean (example)

![2023-08-04_17-42-font]({{site.url}}/images/2023-08-03-Github-Blog-Posting-Font/2023-08-04_17-42-font.png)

- Select Light 300 (example) > select @import > copy clipboard

```html
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300&display=swap');
```



![2023-08-04_17-44-font2]({{site.url}}/images/2023-08-03-Github-Blog-Posting-Font/2023-08-04_17-44-font2.png)

- Go to (local project path)/_sass/minimal-mistakes.scss
- Add below lines

```scss
/* Google fonts */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300&display=swap');
```

- Go to google font and check font family name in 'CSS rules to specify families'

![2023-08-04_17-44-font2]({{site.url}}/images/2023-08-03-Github-Blog-Posting-Font/2023-08-04_17-44-font2-1691138980492-4.png)

- Go to (local project path)/_sass/minimal-mistakes/_variables.scss
- Add font name like below

```scss
/* system typefaces */
$serif: Georgia, Times, serif !default;
$sans-serif: -apple-system, BlinkMacSystemFont, "Noto Sans KR", "Roboto", "Segoe UI",
  "Helvetica Neue", "Lucida Grande", Arial, sans-serif !default;
$monospace: Monaco, Consolas, "Lucida Console", monospace !default;

```



## Change Font Size

- Go to (local project folder)/_sass/_minimal-mistakes/_reset.scss

- Change font-size: as you want


```scss
html {
  /* apply a natural box layout model to all elements */
  box-sizing: border-box;
  background-color: $background-color;
  font-size: 10px;

  @include breakpoint($medium) {
    font-size: 12px;
  }

  @include breakpoint($large) {
    font-size: 14px;
  }

  @include breakpoint($x-large) {
    font-size: 16px;
  }

  -webkit-text-size-adjust: 100%;
  -ms-text-size-adjust: 100%;
}
```






# Reference

- Font 
  - 테디노트 영상 [https://www.youtube.com/watch?v=k7DjQ1JF9rY](https://www.youtube.com/watch?v=k7DjQ1JF9rY)
- Font Size 
  - Blog [https://velog.io/@eona1301/Github-Blog-minimal-mistakes-%EB%B3%B8%EB%AC%B8-%EC%98%81%EC%97%AD-%EB%B0%8F-%EA%B8%80%EC%9E%90-%ED%81%AC%EA%B8%B0](https://velog.io/@eona1301/Github-Blog-minimal-mistakes-%EB%B3%B8%EB%AC%B8-%EC%98%81%EC%97%AD-%EB%B0%8F-%EA%B8%80%EC%9E%90-%ED%81%AC%EA%B8%B0)