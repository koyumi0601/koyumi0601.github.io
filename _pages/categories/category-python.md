---
title: "Python"
layout: archive
permalink: categories/python
author_profile: true
types: posts
---

{% assign posts = site.categories['python']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}