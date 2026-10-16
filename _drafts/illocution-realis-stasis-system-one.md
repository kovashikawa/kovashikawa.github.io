---
layout: single
title: "Illocution, realis and stasis: three ways to read a question"
excerpt: "Three terms from linguistics and rhetoric, what each one asks about a sentence, and why they matter when a fast model has to answer a typed question in one pass."
description: "Illocution, realis and stasis explained with one example sentence, and how they apply to System One models that answer typed questions with probabilities."
tags: [linguistics, rhetoric, llm, system-one, jev, typesafe]
categories: [ai]
author_profile: true
toc: false
read_time: true
header:
  og_image: /assets/images/og/illocution-realis-stasis-system-one.jpg
---

A fast model reads a question once and answers it. Before that happens, someone has to decide what the question is. Linguistics and rhetoric have three old terms for that decision, and they get mixed up because each one sounds like a way of describing a sentence. They describe different things.

I will use one sentence throughout: "I'll send the invoice on Friday."

## Illocution

J. L. Austin split an utterance into three layers. The locution is the words and their meaning. The illocution is the act performed by saying them. The perlocution is the effect on the listener.

The same words can carry different acts. "I'll send the invoice on Friday" can be a promise, a forecast, or a warning to a client who wanted it Thursday. John Searle later sorted these acts into five families: assertions, directives, commitments, expressions of feeling, and declarations. The words alone do not fix the act. The context does.

Illocution asks what the speaker is doing.

## Realis

Realis is a grammatical category. A realis statement presents an event as actual. An irrealis statement presents it as not actual, for instance because it is in the future, possible, wanted, conditional or commanded.

"I sent the invoice on Friday" is realis. "I'll send it," "I might send it," "if I send it" and "send it" are all irrealis. Languages mark the line differently. Some mark it in the verb, and English does it with tense, modals and conditionals.

Realis describes how a statement is presented, not whether it is true. "I sent the invoice" is realis even if no invoice went out.

Realis asks whether this is offered as something that happened.

## Stasis

Stasis comes from classical rhetoric, in the tradition of Hermagoras and Cicero. It names the point where two sides actually disagree. People can argue for an hour without finding it.

The standard questions come in order. Conjecture asks whether the thing happened. Definition asks what it counts as. Quality asks how good, bad or serious it is. The classical lists add a fourth, about where the matter should be decided, and modern teaching often puts policy there, meaning what should be done.

Suppose a client says the invoice never arrived. The first dispute is conjecture: was it sent? If you agree it was, the next is definition: does an email that landed in spam count as sent? After that comes quality: is a day late a breach of the contract? Arguing about quality while the other person is still disputing conjecture is how two people talk past each other.

Stasis asks which question is in dispute.

## Side by side

| Term | Field | Asks | In the invoice example |
|---|---|---|---|
| Illocution | Philosophy of language | What act is being performed? | Promise, forecast or warning |
| Realis | Grammar | Is it presented as actual? | Irrealis, because it is in the future |
| Stasis | Rhetoric | Which question is in dispute? | Sent, counted as sent, or serious |

## What this changes for System One models

System One models such as TypeSafe's Jev do not generate text. They take a state and a set of typed questions and return probabilities. According to [TypeSafe's documentation](https://docs.typesafe.ai/confidence), a Noul is a yes/no question answered with one probability, a Choice picks among unordered options, and a Score places the answer on ordered levels.

My reading, and not something TypeSafe says, is that the three question types line up with the first three stases. A Noul is a conjecture question, a Choice is a definition question, and a Score is a quality question. Picking the type means deciding which dispute you are in.

Illocution and realis show up in what you feed in and how you word the question. Take "I'll send the invoice on Friday" as the state. "Was the invoice sent?" and "Was an invoice promised?" have different right answers over the same text. The first asks about something actual and the second asks about an act. A model that answers in a single pass cannot ask which one you meant. TypeSafe's docs say that flat probabilities often mean the options or levels are ambiguous, so a vaguely worded question tends to show up as an unusable distribution.

The practical use of the three terms is to settle three things before writing the question. Decide what act the text performs, whether it reports something actual, and which dispute you are resolving.
