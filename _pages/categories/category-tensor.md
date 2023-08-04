---
title: "Tensor"
layout: archive
permalink: categories/tensor
author_profile: false
types: posts
sidebar:
  nav: "docs"
---

{% assign posts = site.categories['tensor']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}