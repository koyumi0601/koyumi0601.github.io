---
title: "Physics"
layout: archive
permalink: categories/physics
author_profile: true
types: posts
---

{% assign posts = site.categories['physics']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}