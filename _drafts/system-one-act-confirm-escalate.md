---
layout: single
title: "Thresholds for a System One model: act, confirm or escalate"
excerpt: "Set a confidence threshold per action, from what a mistake costs. What TypeSafe's docs suggest, what decision theory adds, and how confidence relates to probability."
description: "How to set confidence thresholds on a System One model: the docs' three ranges, a cost rule from decision theory, and why confidence is not probability."
tags: [system-one, jev, typesafe, llm, calibration]
categories: [ai]
author_profile: true
toc: false
read_time: true
mathjax: true
header:
  og_image: /assets/images/og/system-one-act-confirm-escalate.jpg
---

TypeSafe's [System One models](https://docs.typesafe.ai/concepts/system-one), of which Jev is the first, return typed decisions and probabilities instead of generated text. The right confidence threshold depends on what a mistake costs, so it should differ per action. Approving a transfer by mistake and showing the wrong balance by mistake do not cost the same.

The three-range pattern below comes from TypeSafe's docs. The cost rule comes from decision theory, which the docs do not cite.

## The pattern in the docs

The [Confidence page](https://docs.typesafe.ai/confidence) offers "a useful starting pattern" of three ranges. High confidence acts automatically. Medium confidence proceeds with caution, which can mean asking the user to confirm or flagging for review. Low confidence does not act, and the docs suggest routing to a human, asking for clarification or falling back. I call these act, confirm and escalate.

The page adds that "a confidence threshold is not one number", since actions should be gated by the consequences of being wrong. In the [voice banking example](https://docs.typesafe.ai/patterns/confidence-routing), anything under 0.6 goes to a support agent. Checking a balance passes at 0.6. Approving a transfer runs on its own only above 0.85, and between 0.6 and 0.85 the system asks the user to confirm. The page describes it as a floor with a threshold per action: "The 0.6 floor catches anything the model is genuinely uncertain about. Above that floor, each action type has its own threshold based on the consequences of acting on a wrong classification."

The Confidence page's own example uses 0.5 and 0.9 instead, and the docs say "The correct threshold values depend on your domain and the performance of the model for your use case." So the numbers are illustrations.

## The rule from decision theory

This part is not TypeSafe's. [Elkan (2001)](https://cseweb.ucsd.edu/~elkan/rescale.pdf) says to predict positive when its expected cost is no higher than the expected cost of predicting negative. Write $C_{FP}$ for the cost of acting when you should not and $C_{FN}$ for the cost of not acting when you should. With no cost for correct decisions, act when $p \ge p^*$, where $p$ is the model's probability that acting is the right call and

$$
p^* = \frac{C_{FP}}{C_{FP} + C_{FN}}
$$

The rule needs $p$ to be a calibrated probability. The docs say TypeSafe trains for calibration, measured across groups of predictions, and that this "does not guarantee that an individual answer is correct." They do not say to plug the probabilities into this rule. That step is mine.

Asking the user is also an action with a cost. Say asking costs $c$ and the user's answer settles the matter, so asking carries no risk of a wrong action, while a wrong action costs $C$. Acting is cheaper when $(1 - p)\,C \le c$, which is $p \ge 1 - c/C$. That is my own algebra from the same principle, not Elkan's formula. With made-up costs, a wrong balance costs 2 and asking costs 1, so act at 0.50 or more. A wrong transfer costs 100 and asking costs 1, so act only at 0.99 or more. The costs are invented, and the cutoffs differ a lot because the error costs do.

[Chow (1970)](https://doi.org/10.1109/tit.1970.1054406) covers the third range. When even the best option's probability is too low, the system rejects and lets a person decide. Where that line sits depends on what a rejection costs compared with an error. My reading is that the docs' routing page has the same shape as these two ideas, a floor for the whole question and then a threshold for each action.

## Confidence and probability

TypeSafe defines Choice confidence, for $n$ options, as

$$
\text{confidence} = \frac{p_{\max} - 1/n}{1 - 1/n}
$$

Inverting it gives $p_{\max} = \text{confidence} \cdot (1 - 1/n) + 1/n$. A confidence of 0.9 is a top probability of 0.95 with two options, 0.9333 with three, and 0.9013 with 75. For a fixed question the two are the same knob on different scales, and the scale depends on the number of options. The docs list the top probability as an alternative to confidence and say its meaning depends on the number of options, so its threshold should be set per question.

The cost rule is stated in probability, so convert confidence to $p_{\max}$ or threshold $p_{\max}$ directly. On a three-option Choice, where the action is the top option, the two cutoffs above of 0.50 and 0.99 are confidences of 0.25 and 0.985. The docs' routing example thresholds confidence, not probability.

A Noul has no confidence field, so it needs no conversion. Its page says where to set the threshold depends on the cost of being wrong: raise it when acting on a false yes is expensive, and lower it when missing a true yes is.

## Setting the thresholds

For each action, write down what a wrong call costs and what asking costs, and set its threshold at $p \ge 1 - c/C$. When to escalate to a person instead of asking is a separate call that I do not formalise here. Then check on your own data that a probability of 0.9 means about 90%.
