---
title: "Setup"
layout: archive
permalink: categories/setup
author_profile: false
types: posts
sidebar:
  nav: "docs"
---

{% assign posts = site.categories['setup']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}