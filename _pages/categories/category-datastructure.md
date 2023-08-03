---
title: "Data Structure"
layout: archive
permalink: categories/datastructure
author_profile: true
types: posts
---

{% assign posts = site.categories['datastructure']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}