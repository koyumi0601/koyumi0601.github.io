---
title: "Book Review"
layout: archive
permalink: categories/book
author_profile: false
types: posts
sidebar:
  nav: "docs"
---

{% assign posts = site.categories['book']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}