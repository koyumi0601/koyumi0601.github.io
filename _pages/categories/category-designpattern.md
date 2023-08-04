---
title: "Design Pattern"
layout: archive
permalink: categories/designpattern
author_profile: false
types: posts
sidebar:
  nav: "docs"
---

{% assign posts = site.categories['designpattern']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}