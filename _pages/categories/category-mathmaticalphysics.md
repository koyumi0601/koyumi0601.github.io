---
title: "Mathmatical Physics"
layout: archive
permalink: categories/mathmaticalphysics
author_profile: false
types: posts
sidebar:
  nav: "docs"
---

{% assign posts = site.categories['mathmaticalphysics']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}