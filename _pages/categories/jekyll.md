---
layout: archive
permalink: jekyll
title: "Jekyll"

author_profile: true
sidebar:
  nav: "docs"
---

{% assign posts = site.categories.jekyll %}
{% for post in posts %}
  {% include archive-single.html type=entries_layout %}
{% endfor %}