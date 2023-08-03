---
title: "Mathmatical Physics"
layout: archive
permalink: categories/mathmaticalphysics
author_profile: true
types: posts
---

{% assign posts = site.categories['mathmaticalphysics']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}