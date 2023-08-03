---
title: "Setup"
layout: archive
permalink: categories/setup
author_profile: true
types: posts
---

{% assign posts = site.categories['setup']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}