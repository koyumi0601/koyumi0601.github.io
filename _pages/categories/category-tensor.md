---
title: "Tensor"
layout: archive
permalink: categories/tensor
author_profile: true
types: posts
---

{% assign posts = site.categories['tensor']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}