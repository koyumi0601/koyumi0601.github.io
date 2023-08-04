---
title: "House Chores"
layout: archive
permalink: categories/housechores
author_profile: false
types: posts
sidebar:
  nav: "docs"
---

{% assign posts = site.categories['housechores']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}