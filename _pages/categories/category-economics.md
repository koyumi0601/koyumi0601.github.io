---
title: "Economics"
layout: archive
permalink: categories/economics
author_profile: true
types: posts
---

{% assign posts = site.categories['economics']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}