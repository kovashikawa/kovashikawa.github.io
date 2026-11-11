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

I used k-means a lot in the past, so e-means caught my attention. It solves the same familiar problem: choose k centroids and minimize the squared distance from each point to its nearest one. The interesting part is how much faster it gets there.[1]

The scale test is the striking part. The authors report clustering 10 billion DINO embeddings into roughly 16.7 million clusters in 2.5 epochs, reaching a slightly lower test loss than standard full-batch k-means at 25 epochs. That is one-tenth as many passes over the training points.[1]

## What changes

Standard full-batch k-means assigns the whole dataset to centroids before updating them. e-means updates throughout each pass, using mini-batches and exponential moving averages of each cluster’s counts and sums. Older assignments gradually lose weight.[1]

Mini-batches and moving averages already existed. The contribution is the recipe around them: a decaying update schedule, improved initialization, revival of underused centroids by splitting active clusters, and a finalization step. Fitted scaling laws choose the batch size and update rate from the dataset size, cluster count, and epoch budget. Specialized GPU kernels turn the algorithmic improvement into runtime savings.[1]

I find that combination interesting. A faster method is much more useful if it does not leave me with another hyperparameter search before I can use it.

## What stays the same

e-means still optimizes the k-means objective. Section 3.1 explicitly says it “assigns each point to its nearest centroid.” Algorithm 1 has no separate noise label.[1]

The geometric limitation follows from that assignment rule. With fixed centroids, nearest-centroid regions are convex. This is still not density-based clustering, and lower loss does not necessarily mean more meaningful categories.

## Try it in the browser

The [project website](https://e-means.github.io) explains the method with animations.[2] There is also an [interactive playground](https://e-means.github.io/playground.html) where you can change the data shape, cluster count, and epoch budgets, then compare the centroid movements and training and test loss. It is a 2D illustration, not a reproduction of the large-scale benchmark; the demo also disables dead-cluster revival for these low-dimensional examples.[3]

## Less unnecessary work

My takeaway is that more work per decision does not automatically buy a better result. Here, it is the same dataset, processed in smaller batches and fewer passes, not evidence that less training data is better.[1]

I see a related design choice in Jev and TypeSafe’s System One models, which return typed decisions and probabilities instead of generating text.[4] The mechanisms are different, but both make me question how much computation a task actually needs.

For clustering, the practical payoff is cheaper reruns. I could compare more choices of k or revisit a dataset after changing its features. The paper cautions that its scaling laws are not optimal for tiny, very low-dimensional datasets, so I would not assume the same gains everywhere.[1] But it gives me a reason to revisit a tool I have used a lot.

## Sources

[1] <https://arxiv.org/html/2611.09501>

[2] <https://e-means.github.io>

[3] <https://e-means.github.io/playground.html>

[4] <https://docs.typesafe.ai/concepts/system-one.md>
