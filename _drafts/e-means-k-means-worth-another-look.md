---
layout: single
author_profile: true
title: "e-means Makes k-means Worth Another Look"
excerpt: "e-means reaches standard k-means quality in far fewer passes. What changes, what stays the same, and why cheaper clustering is worth another look."
description: "e-means reaches standard k-means quality in far fewer passes. What changes, what stays the same, and why cheaper clustering is worth another look."
date: 2026-11-11
last_modified_at: 2026-11-11
categories:
  - ai
tags:
  - clustering
  - k-means
  - e-means
  - machine-learning
header:
  og_image: /assets/images/og/e-means-k-means-worth-another-look.jpg
toc: false
---

I used and researched k-means a lot, so e-means caught my attention. It solves the same familiar problem: choose k centroids and minimize the squared distance from each point to its nearest one. The interesting part is how much faster it gets there.[1]

The authors report clustering 10 billion DINO embeddings into roughly 16.7 million clusters in 2.5 epochs, reaching a slightly lower test loss than standard full-batch k-means at 25 epochs. That is one-tenth as many passes over the training points.[1]

## What changes

Standard full-batch k-means assigns the whole dataset to centroids before updating them. e-means updates throughout each pass, using mini-batches and exponential moving averages of each cluster’s counts and sums. Older assignments gradually lose weight.[1]

Mini-batches and moving averages already existed. The contribution is the recipe around them: a decaying update schedule, improved initialization, revival of underused centroids by splitting active clusters, and a finalization step. Fitted scaling laws choose the batch size and update rate from the dataset size, cluster count, and epoch budget. Its GPU kernels keep each pass efficient, so fewer passes translate into runtime savings.[1]

The automatic settings matter to me. I don't want to trade the time saved on training for another hyperparameter search.

## What stays the same

e-means still optimizes the k-means objective. Section 3.1 explicitly says it “assigns each point to its nearest centroid.” Algorithm 1 has no separate noise label.[1]

The geometric limitation follows from that assignment rule. With fixed centroids, nearest-centroid regions are convex. This is still not density-based clustering, and lower loss does not necessarily mean more meaningful categories.

## Try it in the browser

The [project website](https://e-means.github.io) explains the method with animations.[2] There is also an [interactive playground](https://e-means.github.io/playground.html) where you can change the data shape, cluster count, and epoch budgets, then compare the centroid movements and training and test loss. It is a 2D illustration, not a reproduction of the large-scale benchmark; the demo also disables dead-cluster revival for these low-dimensional examples.[3]

## Why I care

For me, the useful part is being able to try more values of k or rerun the analysis after changing the features. A faster run gives me more room to experiment.

Jev, TypeSafe's System One model, interests me for a related reason. It returns typed decisions and probabilities rather than generated text.[4] If my code needs a decision, I don't need a paragraph explaining it. The connection I see is practical, not a claim that the methods work the same way.

e-means still uses the same dataset; it needs fewer passes.[1] The authors caution that their scaling laws are not optimal for tiny, very low-dimensional datasets, so I would not assume the same gains everywhere.[1] Still, it makes me want to revisit some of my old work.

## Sources

[1] <https://arxiv.org/html/2611.09501>

[2] <https://e-means.github.io>

[3] <https://e-means.github.io/playground.html>

[4] <https://docs.typesafe.ai/concepts/system-one.md>
