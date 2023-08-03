---
title: "House Chores"
layout: archive
permalink: categories/housechores
author_profile: true
types: posts
---

{% assign posts = site.categories['housechores']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}