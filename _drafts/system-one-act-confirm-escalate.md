---
layout: single
title: "Thresholds for a System One model: act, confirm or escalate"
excerpt: "Set a confidence threshold per action, from what a mistake costs. What TypeSafe's docs suggest, what decision theory adds, and how confidence relates to probability."
description: "How to set confidence thresholds on a System One model: the docs' three ranges, a cost rule from decision theory, and how confidence maps to the top probability."
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

The Confidence page's own example uses 0.5 and 0.9 instead, and still asks for confirmation above 0.9. The docs say "The correct threshold values depend on your domain and the performance of the model for your use case." So the numbers are illustrations.

## The rule from decision theory

This part is not TypeSafe's. The principle is old and simple, and [Elkan (2001)](https://cseweb.ucsd.edu/~elkan/rescale.pdf) states it for classifiers: take the action whose expected cost is lowest. [Chow (1970)](https://doi.org/10.1109/tit.1970.1054406) covers refusing to decide. His rule rejects when the largest probability is too low, and the cutoff is set by what a rejection costs compared with what an error costs.

Here asking the user is the rejection. Say asking costs $c$ and the user's answer settles the matter, so asking carries no risk of a wrong action, while a wrong action costs $C$. Acting is cheaper when $(1 - p)\,C \le c$, so act when

$$
p \ge 1 - \frac{c}{C}
$$

where $p$ is the probability of the option the action carries out, and the only way to be wrong is that option being wrong. This is Chow's reject rule, with one cost for all errors, a correct decision free, and the costs written as $c$ and $C$. With made-up costs, a wrong balance costs 2 and asking costs 1, so act at 0.50 or more. A wrong transfer costs 100 and asking costs 1, so act only at 0.99 or more. The costs are invented, and the cutoffs differ a lot because the error costs do.

The rule needs $p$ to be a calibrated probability. The [System One page](https://docs.typesafe.ai/concepts/system-one) says these models are trained for calibrated decisions, measured across groups of predictions, and that this "does not guarantee that an individual answer is correct." The docs do not say to plug the probabilities into this rule. That step is mine.

The rule compares acting with asking only. It also assumes the user's answer settles the case. When $p$ is low, the top option is often not what the user meant, a "no" leaves nothing to act on, and that is where a person comes in.

## Confidence and probability

TypeSafe defines Choice confidence, for $n$ options, as

$$
\text{confidence} = \frac{p_{\max} - 1/n}{1 - 1/n}
$$

Inverting it gives $p_{\max} = \text{confidence} \cdot (1 - 1/n) + 1/n$. A confidence of 0.9 is a top probability of 0.95 with two options, 0.9333 with three, and 0.9013 with 75. For a fixed question the two are the same knob on different scales, and the scale depends on the number of options. The docs list the top probability as an alternative to confidence and say its meaning depends on the number of options, so its threshold should be set per question.

The cost rule is stated in probability, so convert a Choice's confidence to $p_{\max}$ or threshold $p_{\max}$ directly. On a three-option Choice, where the action is the top option, the two cutoffs above of 0.50 and 0.99 are confidences of 0.25 and 0.985. The docs' routing example thresholds confidence, not probability.

## Setting the thresholds

For each action, write down what a wrong call costs and what asking costs, and set its threshold at $p \ge 1 - c/C$. Then check on your own data that a probability of 0.9 means about 90%.
