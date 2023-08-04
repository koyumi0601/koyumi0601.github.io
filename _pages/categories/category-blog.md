---
title: "Blog"
layout: archive
permalink: categories/blog
author_profile: false
types: posts
sidebar:
  nav: "docs"
---

{% assign posts = site.categories['blog']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}