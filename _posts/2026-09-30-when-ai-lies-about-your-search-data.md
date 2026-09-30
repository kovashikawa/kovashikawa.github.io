---
layout: single
title: "When AI Lies About Your Search Data"
date: 2026-09-30 13:52:00 +0200
excerpt: "The long version of my Search Central Live Deep Dive talk: why the dangerous AI error is the fluent one, and the three guardrails we run in production to make numbers prove where they came from."
description: "How FUSE's Emet pipeline checks AI answers about search data: declared receipts, a deterministic verifier whose tolerance comes from how a number was written, and a judge from a different model provider, with the research behind each choice."
tags: [AI, LLM, hallucination, verification, search-console, agents]
categories: [ai]
author_profile: true
toc: true
toc_sticky: true
read_time: true
mathjax: true
---

On September 30 I gave a seven-minute lightning talk at Google Search Central Live Deep Dive Europe 2026 in Barcelona. Seven minutes leaves room for the argument, but not for the papers, the edge cases, or more than a sentence on the time our own agent broke. This post is the long version.

"Lie" is the talk's title and my shorthand. A model has no intent, but the reader is misled all the same. The core problem is simple to state: a language model gives you a plausible number, not necessarily a correct one, and the screen looks identical either way. At FUSE we don't take the model's word for a number. We ask each number to show where it came from.

## A plausible sentence with the wrong verb

Picture a chatbot summarizing last month's Search Console Performance export:

> "12,480 organic clicks in August."
>
> "Average position improved, rising from 8.2 to 9.1."

The first claim agrees with the export. The second pairs two correct figures with the wrong verb. Google defines position as ["a relative ranking of the position of your link on Google, where 1 is the topmost position"](https://support.google.com/webmasters/answer/7042828), averaged over impressions, so a move from 8.2 to 9.1 means the site showed up further from the top on average. "Rising" is slippery even among practitioners: Google's own [bubble-chart guide](https://developers.google.com/search/docs/monitor-debug/bubble-chart-analysis) puts 1 at the top of its chart and talks about a low-ranking query's average position that "moves up". Read "rising" either way and "improved" is still wrong.

Two caveats, because Search Console people will raise them.

First, a worse average position isn't automatically bad news. Position is only recorded when there's an impression, so a site that starts appearing for new, lower-ranking queries can see its average position number go up while it gains visibility. Google's help calls position ["a complex metric that can be misleading if you don't understand the subtleties"](https://support.google.com/webmasters/answer/7042828) and [recommends](https://support.google.com/webmasters/answer/17010961) focusing more on trends in impressions and clicks. The error in the example is the direction word, not a business judgment.

Second, this isn't a claim that frontier models make this exact mistake often. A strong model on a high reasoning setting may well catch it. The interesting part isn't the frequency. It's what the failure looks like when it happens: fluent, confident, and silent.

Pascal has a line for this. In a note headed "Langage" in the *Pensées* (Brunschvicg 27, Lafuma 559), he compares writers who force words into neat antitheses to builders who add false windows for symmetry. In Rudolf Arnheim's rendering in *Entropy and Art*: "their rule is not to speak right but to make right figures." Pascal meant figures of speech. The English word also means numbers, a sense he never intended, and read that way the line describes the example exactly. The figures, 8.2 and 9.1, were right. The speaking was not. "Improved" is a false window: it's there for the shape of the sentence, with nothing behind it. Arnheim's own gloss, just before the quote, is the best one-line summary of the problem I know: "The form may be quite orderly and yet misleading, because its structure does not correspond to the order it stands for."

## Why fluent and wrong is the default

This isn't a quirk of one model. Part of it is how models are trained and scored.

Training and evaluation ["reward guessing over acknowledging uncertainty"](https://arxiv.org/abs/2509.04664). It works like a multiple-choice exam: a guess might score, a blank never does. Human feedback can push the same way. People, and the preference models trained on their ratings, [sometimes prefer a convincing answer over a correct one](https://arxiv.org/abs/2310.13548), and when OpenAI [rolled back a GPT-4o update](https://openai.com/index/sycophancy-in-gpt-4o/) that had become too agreeable, it [wrote](https://openai.com/index/expanding-on-sycophancy/) that user feedback "can sometimes favor more agreeable responses."

More capable models help, but capability isn't reliability. On [AA-AnalystAgent](https://artificialanalysis.ai/articles/aa-analyst-agent), a benchmark of hard analyst tasks, the top model at launch averaged 64 percent on a single attempt but was right on all five attempts on only 54 percent of them. And when a task leaves a definition out, agents often don't ask. On [Ambig-DS](https://arxiv.org/abs/2605.09698), models that weren't told which metric to optimize often failed quietly: they picked a plausible but wrong one, or handed in a baseline.

None of this means models lie. Part of the signal rewards sounding right, and a chat window renders a right answer and a wrong one in exactly the same way.

## Errors travel, corrections don't

Before FUSE I spent more than three years at a hedge fund building inflation forecasting models, with a lot of Eurostat data and some from the UK's Office for National Statistics along the way. What stuck with me is that an inflation print is an input. It feeds forecasts, positions and risk limits, and nothing downstream goes back to re-read the source.

The ONS gave a clean example after I'd left. On May 21, 2025 it published April's annual CPI inflation rate at 3.5 percent. On June 5 it [said](https://www.ons.gov.uk/news/statementsandletters/vehicleexcisedutyimpactonconsumerpriceinflation) that faulty vehicle excise duty data from the Department for Transport had overstated the rate by 0.1 percentage points, so the correct figure was 3.4 percent. In line with its existing policy, the ONS didn't amend the published figure. Its [current revisions policy](https://www.ons.gov.uk/economy/inflationandpriceindices/articles/revisionsandcorrectionoferrorspoliciesforconsumerpriceinflationstatistics/december2025) explains why: index-linked gilts, pensions, rail fares and commercial contracts all rely on the published numbers. RPI was also 0.1 percentage points too high, 4.5 instead of 4.4, and RPI sets payments on Britain's index-linked gilts. A Deutsche Bank economist who had forecast 3.4 [told Reuters](https://www.reuters.com/world/uk/uk-car-tax-error-overstated-inflation-by-10-basis-points-ons-says-2025-06-05/) that "a lot of clients have lost a lot of money." The ONS owned up about two weeks later, and the official April figure still reads 3.5.

Search data has the same shape. An average position becomes a line in a report, the report becomes a budget, and the budget becomes a decision. Nobody in that chain goes back to the rows.

Agents add a compounding argument, with caveats. The usual illustration is multiplicative: if $n$ steps must all succeed and each succeeds with probability $p$, the whole run succeeds with probability $p^n$, and at $p = 0.99$ that falls to about one half by $n = 69$ ($0.99^{69} \approx 0.4998$). It's the step-level version of the $(1-e)^n$ argument Yann LeCun [made per token](https://www.linkedin.com/posts/yann-lecun_i-have-claimed-that-auto-regressive-llms-activity-7045908925660950528-hJGk), and it assumes independent steps and no recovery. Real chains break that assumption in both directions. A popular constant-hazard model of agent failure was [revised by its own author in 2026](https://www.tobyord.com/writing/hazard-rates-for-ai-agents-decline) after a reanalysis found that hazard rates fall as a task goes on: long tasks go better than the simple model predicts, and short tasks held to high reliability go worse. Three empirical findings matter more:

- **Self-conditioning.** Models [become more likely to make mistakes](https://arxiv.org/abs/2509.09677) once their own earlier mistakes are in the context. The thinking models tested resisted the effect.
- **Snowballing.** Once a model commits to a wrong answer, it [can go on to justify it](https://proceedings.mlr.press/v235/zhang24ay.html) with claims it would itself flag as wrong if asked about them separately. GPT-4 caught 87 percent of such claims on its own.
- **Numbers turn into prose.** In [a 2026 study](https://arxiv.org/abs/2608.14588), numeric errors planted early in a four-agent finance pipeline had become narrative by the last stage, and GPT-4o's detection fell from 72 to 51 percent. Checking at every handoff left 16 percent of the errors alive, against 58 percent with a single check at the end.

The practical conclusion: check a number while it's still a number.

## What happened to our own agent in January

We learned the other half of the context lesson the hard way: what's missing from an agent's context matters as much as what's in it. Around the turn of the year, a refactor split our Search Console agent into two steps, fetch and then analyze. The prompt it ran with in January told it, in part:

```
Your script's ONLY job is to retrieve the right data. Keep it simple:
[...]
DO NOT add any of these - they are handled automatically by the system:
- Summary calculations (totals, averages, percentages)
[...]
- CTR calculations or conversions
```

The same prompt assumed a later step would "generate a detailed summary with insights". That step wasn't there. The harder questions also went down a path that no longer loaded the field reference or the worked examples. The rows it fetched were real. The analysis built on them had lost the context it needed.

The first fix was more code, a separate analyze step, and it didn't hold up. The fix that worked was mostly not code: we let the script compute again and put the context back, the field reference and more worked examples. The lesson fits on one line: the brains of an AI agent are in its context, not its code. One detail is telling. The January prompt held up `df['ctr_pct'] = df['ctr'] * 100` as an example of an over-engineered script. Today the same line is the worked example under rule one of the agent's Search Console rules, which says CTR is a fraction, not a percentage.

Restoring the context restored the quality of the analysis. Context alone doesn't make output verifiable, so we built a system around the model.

## Emet: one rule, three guardrails

Emet, from the Hebrew word for truth, runs on FUSE's production traffic. It rests on one rule: a model never grades its own work. The work splits into three guardrails.

### 1. Declare: every number gets a receipt

When the agent presents numbers from tool data, it's instructed to end its answer with a structured receipt that's stripped from the text the reader sees. Each entry carries the value as written, the metric, where it came from, and, for a derived number, the formula:

```
ctr                2.42%    row ctr 0.0242, from the API
ctr_rel_change    +14.7%    (0.0242 - 0.0211) / 0.0211 * 100
```

The second line is also a small lesson in wording. CTR moving from 2.11 to 2.42 percent is a 14.7 percent relative change and a 0.31 point absolute change, and an answer that says "CTR rose 14.7 percent" should say which one it means.

### 2. Verify: plain code, no model

Every figure the model says it read from a data tool is checked against the tool output saved when the answer was written. Not the model's word, and not a fresh query: sources move (Search Console's newest data [can be preliminary](https://support.google.com/webmasters/answer/7576553)), and the question here is whether the model was faithful to its evidence.

The design question is how close counts as a match. A flat percentage tolerance scales with the number: 5 percent of 10,000 clicks is 500 clicks of slack. Emet doesn't use one. The tolerance comes from how the number was written. "12,480" has to match to the click, while "12K" only claims thousands, so anything that rounds to 12K can match. A figure written too vaguely to prove much doesn't get a checkmark for landing somewhere near the truth. It's the same rule metrologists use for a digital reading ([GUM](https://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e_FILES/MAIN_C045315e/AF_e.html), F.2.2.1) and financial reporting uses for rounded facts ([XBRL Calculations 1.1](https://www.xbrl.org/Specification/calculation-1.1/REC-2023-02-22/calculation-1.1-REC-2023-02-22.html), section 5.2.3).

It also never guesses units. Search Console's API returns CTR as a fraction from 0 to 1 while the report shows a percent, and a comparator can't tell a fraction from a hundredfold error unless the unit is written down. So a mismatch is never quietly converted into a match.

No match, no checkmark.

### 3. Judge, then annotate

The plain-code check covers numbers. Much of what misleads is in the words around them: directions, rankings, comparisons. For those, one model extracts the factual claims the answer rests on (values, directions, rankings, counts and dates), and a second model, chosen from a different provider than the one that wrote the answer, grades each claim against the saved rows as supported, imprecise, or unsupported. A wrong direction makes a claim unsupported.

The different provider isn't decoration. LLM judges [recognize and favor their own generations](https://arxiv.org/abs/2404.13076), and [a panel of models from different families](https://arxiv.org/abs/2404.18796) showed less intra-model bias than a single large judge. We run one grader, not a panel. What we take from both papers is to keep it out of the writer's family.

Because the grader is itself a language model, it proposes rather than overwrites: an unsupported claim gets a correction in the verification panel under the answer, with the original struck through, the corrected text, and the reason, one click away. The answer text itself is never altered. It runs in the background, so the answer isn't held up while it's checked.

## What it doesn't do

- **It checks faithfulness to the rows, not the truth of the rows.** Emet wouldn't have caught the ONS error, because the input itself was wrong. That's a different tier of problem.
- **Corrections are proposals.** They sit under the answer, and the answer is never rewritten.
- **A number the model never declares never reaches the deterministic check.** Closing that gap is next.

## The part you can steal

Part of this works without any pipeline. Metric semantics are context, not code, and a model can only use the context you give it. For Search Console, four lines cover the most common traps:

```
position  1 is the top. A bigger number is further down.
ctr       the API returns 0 to 1; the report shows a percent. Keep both forms.
clicks    are not sessions. Search Console is not GA4.
combine   sum clicks and impressions. Never average CTR or position
          across rows: CTR = total clicks / total impressions, and
          position is weighted by impressions.
```

Each line rests on Google's own documentation: [position](https://support.google.com/webmasters/answer/7042828), [CTR in the API](https://developers.google.com/webmaster-tools/v1/searchanalytics/query), and [clicks versus sessions](https://developers.google.com/search/docs/monitor-debug/google-analytics-search-console) ("Clicks and sessions are calculated differently").

Write them down once, where your assistant will find them. In Claude, paste them into a chat and ask it to turn them into a [skill](https://support.claude.com/en/articles/12512180-use-skills-in-claude): a recipe card it pulls out when a Search Console export shows up. Skills need code execution turned on, and [Agent Skills](https://agentskills.io) is an open format other tools read too. Project instructions work more like a sticky note it reads in every chat. In Gemini, a Gem's instructions do the same job, and so do custom instructions in ChatGPT.

One habit needs no setup at all: request the source rows for each figure and compare them with an export you pulled yourself. Asking the model whether it got it right is letting it grade its own work.

## Show the seams

In a journal entry from 1843, Kierkegaard grants that life must be understood backward but warns that it must be lived forward. Generation is the forward half. A model writes one token after another, and although the rows are in its context the whole time, nothing in that process goes back to test a sentence against them once it's written. Verification has to supply the other half, tracing each claim back to the rows it rests on.

Building this has moved my trust from the model to the evidence. It has made me insist that every figure it declares can be traced to a row, and that the reader can see what was checked.

## Further reading

1. Kalai, Nachum, Vempala, Zhang (2025). [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664). *arXiv:2509.04664*.
2. Sharma et al. (2023). [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548). *ICLR 2024*.
3. OpenAI (2025). [Sycophancy in GPT-4o: what happened and what we're doing about it](https://openai.com/index/sycophancy-in-gpt-4o/) and [Expanding on what we missed with sycophancy](https://openai.com/index/expanding-on-sycophancy/).
4. Artificial Analysis (2026). [Announcing AA-AnalystAgent: an agentic benchmark for quantitative analysis on real-world spreadsheets and documents](https://artificialanalysis.ai/articles/aa-analyst-agent).
5. Stoisser, Boubnovski Martell, Boldsen, Märtens, Kitchen (2026). [Ambig-DS: A Benchmark for Task-Framing Ambiguity in Data-Science Agents](https://arxiv.org/abs/2605.09698). *arXiv:2605.09698*.
6. Sinha, Arun, Goel, Staab, Geiping (2025). [The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](https://arxiv.org/abs/2509.09677). *ICLR 2026*.
7. Zhang, Press, Merrill, Liu, Smith (2024). [How Language Model Hallucinations Can Snowball](https://proceedings.mlr.press/v235/zhang24ay.html). *ICML 2024*.
8. Singh, Pawar (2026). [The Hallucination Snowball: Modeling Error Propagation as State Transitions in Multi-Agent LLM Pipelines](https://arxiv.org/abs/2608.14588). *FAGEN Workshop, ICML 2026*.
9. LeCun (2023). [LinkedIn post on auto-regressive LLMs](https://www.linkedin.com/posts/yann-lecun_i-have-claimed-that-auto-regressive-llms-activity-7045908925660950528-hJGk), March 27, 2023.
10. Ord (2026). [Hazard Rates for AI Agents Decline as a Task Goes On](https://www.tobyord.com/writing/hazard-rates-for-ai-agents-decline).
11. Panickssery, Bowman, Feng (2024). [LLM Evaluators Recognize and Favor Their Own Generations](https://arxiv.org/abs/2404.13076). *NeurIPS 2024*.
12. Verga et al. (2024). [Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models](https://arxiv.org/abs/2404.18796). *arXiv:2404.18796*.
13. JCGM (2008). [Evaluation of measurement data: Guide to the expression of uncertainty in measurement (GUM)](https://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e_FILES/MAIN_C045315e/AF_e.html), annex F.2.2.1.
14. XBRL International (2023). [Calculations 1.1](https://www.xbrl.org/Specification/calculation-1.1/REC-2023-02-22/calculation-1.1-REC-2023-02-22.html), section 5.2.3.
15. Office for National Statistics (2025). [Vehicle Excise Duty impact on Consumer Price Inflation](https://www.ons.gov.uk/news/statementsandletters/vehicleexcisedutyimpactonconsumerpriceinflation).
16. Google Search Console Help. [What are impressions, position, and clicks?](https://support.google.com/webmasters/answer/7042828)
17. Google Search Central. [Using Search Console and Google Analytics data for SEO](https://developers.google.com/search/docs/monitor-debug/google-analytics-search-console).
18. Pascal, *Pensées*, Brunschvicg 27 / Lafuma 559 ([text and commentary](https://www.penseesdepascal.fr/XXIII/XXIII13-approfondir.php)). Arnheim, R. (1971). *Entropy and Art: An Essay on Disorder and Order*. University of California Press.
