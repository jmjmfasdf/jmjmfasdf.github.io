---
layout: page
title: Gallery
subtitle: From the pexels folder
permalink: /gallery/
feature-img: "assets/img/portfolio/portfolio.png"
gallery_path: "assets/img/pexels"
position: 2
tags: [Page]
---

This is a photo gallery made from the static files in the `assets/img/pexels` folder. 
I wanted to automatically create a simple gallery from a folder without having to create a markdown page as you would for the portfolio.


{% include gallery.html gallery_path=page.gallery_path %}
