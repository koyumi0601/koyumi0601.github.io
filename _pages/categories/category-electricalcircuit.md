---
title: "Electrical Circuit"
layout: archive
permalink: categories/electricalcircuit
author_profile: true
types: posts
---

{% assign posts = site.categories['electricalcircuit']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}