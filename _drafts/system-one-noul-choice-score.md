---
layout: single
title: "Noul, Choice or Score: picking a System One question type"
excerpt: "A System One model returns probabilities, not text. The question type sets their shape, and Choice and Score summarize uncertainty differently."
description: "Noul returns one probability, Choice one per option, Score one per ordered level. A worked example shows where Choice and Score confidence differ."
tags: [system-one, jev, typesafe, llm]
categories: [ai]
author_profile: true
toc: false
read_time: true
header:
  og_image: /assets/images/og/system-one-noul-choice-score.jpg
---

A System One model does not write text. You give it a state, which is the content to judge, and typed questions about that state. It returns probabilities. Jev, TypeSafe's model, is my example. The descriptions of the question types come from TypeSafe's public docs, and sentences that start with "I" or "my" are mine.

The question type fixes the shape of the answer. Choice and Score both return a distribution, so they can look interchangeable, but they measure confidence differently, because a Score's levels have an order.

## What each type returns

A Noul is a yes/no question. The answer is a single number, the probability of yes. There is no separate confidence field, because with two outcomes that one number already describes the whole distribution.

A Choice picks one option from a set with no order between them. The answer holds the chosen option, a probability for every option, and a confidence value.

A Score places the state on ordered levels that you describe in words. The answer holds a score, a legend that maps level numbers back to your descriptions, a probability for every level, and a confidence value. Each level is judged alone against the state, and the order lives only in the numbering. The score is the probability-weighted mean of the level numbers. In the docs' bug severity example, probabilities of 0, 0.57 and 0.43 over three levels give 1.43, a position between levels 1 and 2.

## Choosing a type

The docs choose by the shape of the answer. Choice fits a known set of options with no order, such as a department. Score fits a spectrum where you can describe each point, such as severity. Noul fits a clean yes/no question where the probability itself is the useful signal. The docs also warn against using a Noul for a degree. A Noul of 0.5 means yes and no got equal probability. It does not mean a medium skill level.

The entity alignment cookbook grades a pair of product records as different, related or the same. It uses a Score, and says a Choice "would lose the ordered relationship of the three outcomes".

The docs add a test for a Score. Its levels should sit on one dimension. If there is no in-between and the answer is one of a few discrete categories, they say to use a Choice, or to split the question into several Nouls. I keep the full probabilities for a Choice or a Score, because the chosen option or the score alone hides how the model split.

## Choice and Score confidence

The Confidence page gives each type its own formula. For a Choice with n options, confidence measures the top probability above an even split, `(p_max - 1/n) / (1 - 1/n)`. Only the top probability counts. A Score uses the distances between levels instead. It measures how far the rest of the probability sits from the most likely level, and compares that with a flat spread across all levels.

The docs' worked example uses three levels. I ran both formulas in Python on these two inputs and got the values the docs print.

| Probabilities over three levels | Choice formula | Score formula |
|---|---|---|
| (0, 0.5, 0.5) | 0.25 | 0.25 |
| (0.5, 0, 0.5) | 0.25 | 0 |

In the first row the model is torn between two neighbouring levels. In the second it is torn between the two ends, and Score confidence drops to zero, the formula's floor. The Choice formula gives both rows 0.25 because it sees only the top probability, 0.5 in each.

The second row also shows why the score alone is not enough. Its score is 1.0, the same as a model that put everything on level 1, yet it gave level 1 no probability. The Score page says to read `probabilities` and `confidence` next to the score, and calls a fractional score a position you can rank by or round. The Jev 1.13 notes add that this version's score levels are weak in numerical calibration, so they advise against using a score to rebuild an exact number between two levels.

For a Noul, the docs offer `|2p - 1|` as a confidence-style number. It is the Choice formula for the options (p, 1 - p).

Sources: TypeSafe docs on [Primitives](https://docs.typesafe.ai/primitives), [Score](https://docs.typesafe.ai/primitives/score), [Confidence](https://docs.typesafe.ai/confidence), the [entity alignment cookbook](https://docs.typesafe.ai/cookbooks/entity_alignment) and the [Jev 1.13 notes](https://docs.typesafe.ai/model-jaggedness/jev-1.13).
