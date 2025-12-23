---
title: Index
---

# Writing & Notes

A growing collection of notes on machine learning
written from first principles.

---

## Machine Learning
{% for post in site.categories.ml %}
- [{{ post.title }}]({{ post.url }})
{% endfor %}

## Mathematics
{% for post in site.categories.math %}
- [{{ post.title }}]({{ post.url }})
{% endfor %}

## Systems
{% for post in site.categories.systems %}
- [{{ post.title }}]({{ post.url }})
{% endfor %}
