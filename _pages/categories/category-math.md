---
title: "Mathmatics"
layout: archive
permalink: categories/mathmatics
author_profile: false
types: posts
sidebar:
  nav: "docs"
---

{% assign posts = site.categories['mathmatics']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}