---
title: "Numerical Method"
layout: archive
permalink: categories/numericalmethod
author_profile: false
types: posts
sidebar:
  nav: "docs"
---

{% assign posts = site.categories['numericalmethod']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}