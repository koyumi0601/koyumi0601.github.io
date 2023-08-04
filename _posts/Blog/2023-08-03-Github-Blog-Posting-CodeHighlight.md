---
layout: single
title: "How to fix code highlight(python)"
categories: blog
tags: [Github, Blog, Installation]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

<s> 동작성 확인을 하지 못하였으나 어느새 동작 함 </s>
## Code highlight Failure (Python)
- template doesn't support python code highlight
- Install ruby and plug-in(jekyll-pygments)
### steps:
- Install ruby 
```bash
sudo apt install ruby-bundler 
# sudo snap install ruby --classic
# sudo apt-get install ruby-dev
sudo bundle install
sudo gem update bundler # dependency
```
- Check gem(jekyll-pygments) is installed already
```bash
gem list
```

- Go to project file and generate Gemfile. File this exists, add lines like below
```
source "https://rubygems.org"
gem "jekyll"
gem "jekyll-pygments" # jekyll-pygments 플러그인 추가
```
- Go to project file and install plug-in
```bash
bundle install
```


- Result

```py
a = 1
print(a)
```

```python
a=1
```

```yml
a=1
```

```html
<div> 
    a = 1
</div>
```
