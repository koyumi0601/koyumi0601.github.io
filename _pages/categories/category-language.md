---
title: "Language"
layout: archive
permalink: categories/language
author_profile: false
types: posts
sidebar:
  nav: "docs"
---

{% assign posts = site.categories['language']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}