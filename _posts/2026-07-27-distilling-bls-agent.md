---
title: "The Measurement Was Harder Than the Model"
excerpt: "I distilled a 1.7B model into a BLS tool-calling agent. Six measurement mistakes in a row, a val-loss trap that was a config bug, a fix that came from changing what the model says rather than how it trains, and a lesson about what happens when your test failures and your usability requirements overlap."
date: 2026-07-27
last_modified_at: 2026-07-29
categories:
  - ai
  - projects
tags:
  - distillation
  - lora
  - mlx
  - mcp
  - tool-calling
  - qwen
  - evaluation
classes: wide
toc: true
toc_label: "On this page"
toc_icon: "brain"
---

## The idea

Take Qwen3-1.7B, give it a set of BLS economic data tools, and distill it into a
specialist small enough to run on an M4 Mini. Ask it "what happened to food
prices since 2021?" and it should emit `get_series(series_id="CUUR0000SAF1",
start="2021")`.

The pipeline: 205 hand-written seed questions over 82 economic concepts, 229
training examples, MLX LoRA, a 67MB adapter, six tools.

That part worked. Everything I initially believed about *how well* it worked was
wrong, in several separate ways, and finding each one required disproving the
previous one. The measurement was harder than the model. The fix, when it
finally came, was not a hyperparameter but a change to what the model was asked
to say.

## Act 1: the number that wasn't real

Round one looked clean. Train loss fell from 3.4 to 0.027. Fifteen held-out
examples, 87% accuracy. I fixed some data issues, retrained, got 93%. Base model
scored roughly zero. Ship it.

Then I ran a separate adversarial review of the pipeline, and none of it survived.

**The held-out set was not held out.** I had split the seeds before expanding
them, which sounds right. But the expander drew from shared pools (a list of
series IDs, a list of search terms) regardless of which split a seed belonged
to. So a training seed and a test seed could independently emit byte-identical
rows. 34% of my validation set appeared verbatim in training. Worse, the eval
file itself was stale: it was the *first* round's holdout, and the second round's
reshuffle had moved all 15 of those questions into training. The 93% was
measured on training data.

**The metric was the easy half.** "Accuracy" meant *did it pick the right tool*:
a 6-way classification. Whether the emitted call was actually correct was never
scored.

**A ninth of the labels were wrong.** Eleven series IDs named the wrong concept.
`CUUR0000SETG01` was mapped to "energy"; the BLS catalog calls it **airline
fares**. `CUUR0000SAH1` was "housing" but means *shelter*; `CUUR0000SAM1` was
"medical care" but means *medical care commodities*. Four IDs (`SAR1`, `SAC1`,
`SAS1`, `SEHA01`) do not exist in any BLS catalog at all. I had invented them by
pattern-matching the real ones.

Honest number, measured properly: **13.3% exact match.**

The lesson I'd draw is not "be careful." It's that *every one of these bugs
pushed the number up*. Nobody investigates a pleasant surprise as hard as a
disappointing one, and that asymmetry is the whole problem.

## Act 2: testing the impossible

Fixing the leak exposed a deeper design error.

The split held out whole *concepts*. If "medical care" appeared only in the test
set, the model was being asked to produce `CUUR0000SAM` having never once seen
that mapping. Four of eleven scored items were unanswerable by construction.

This is a lookup task. `"medical care" → CUUR0000SAM` cannot be derived from
first principles; it can only be recalled. So the split should hold out
**phrasings**, not concepts:

- Every concept contributes at least one phrasing to training
- Remaining phrasings go to val/test
- The build **fails** if any concept is held out entirely

I rewrote the seed data as concept tables (82 concepts, each with 2+ distinct
phrasings) and added build-time assertions: every series ID must exist in the
bundled 8,103-row catalog, no example may appear in two splits, no concept may be
missing from train. Assertions, not intentions. Two of them have fired on me
since.

## Act 3: the discovery

With clean data I swept checkpoints from 100 to 1400 iterations, planning to pick
the lowest validation loss like every tutorial says.

Val loss bottomed at iteration 250 and rose steadily afterward. Textbook
overfitting. But I was also scoring actual tool-call accuracy, and the two
disagreed violently:

| iter | val loss | exact match |
|------|---------|-------------|
| 200 | **0.135** (min) | 37.2% |
| 400 | 0.135 | 65.1% |
| 600 | 0.147 | **90.7%** |
| 800 | 0.156 | 81.4% |
| 1000 | 0.169 | 88.4% |
| 1400 | 0.169 | 74.4% |

Stopping at the val-loss minimum would have shipped a 37% model instead of a 91%
one. I wrote this up as a finding: *val loss is a trap for structured-output
tasks.* Cross-entropy punishes a confidently-wrong series ID exactly as hard as
gibberish, while a task evaluator sees a near-miss. I had citations lined up.

It was a config bug.

## Act 4: disproving my own finding

Before publishing I checked one number I had never looked at: the ratio of
completion length to prompt length.

```
mean prompt tokens     : 134
mean completion tokens :  19
generation ratio       : 0.141
```

Every training row carried a ~120-token system prompt, byte-identical across the
dataset, and a ~19-token tool call. And I was training with prompt loss
**unmasked**.

**87.7% of every gradient was the model re-predicting a fixed preamble.**

That explains all of it. Train loss of 0.027 was never impressive: most of it
was copying a constant. And validation loss was ~88% a measurement of *preamble
reproduction*, which is uncorrelated with whether the tool call is right. The two
curves weren't in tension for any deep reason. One of them was mostly noise.

This is documented. Huerta-Enochian and Ko (2024) found a statistically
significant effect of prompt-loss weight specifically for short-completion data,
and Vaughn's walkthrough of the same phenomenon on a multiple-choice dataset
(generation ratio 0.01) reports that stopping at the full-sequence val-loss
minimum yields 53% accuracy while completion loss is still falling. I reproduced
a known failure mode and mistook it for a discovery.

So I masked the prompt and reran. Prediction: val loss should re-couple with
accuracy.

| iter | val loss (masked) | exact match |
|------|------------------|-------------|
| 200 | 0.046 | 62.8% |
| 400 | 0.028 | 86.0% |
| 600 | **0.027** (min) | **90.7%** |
| 800 | 0.028 | 90.7% |

| | unmasked | masked |
|---|---|---|
| pick by val-loss minimum | 37.2% | 86.8% |
| best checkpoint available | 90.7% | 88.4% |
| **penalty for trusting val loss** | **~50 pts** | **1.6 pts** |

The trap was self-inflicted. Mask the prompt and standard practice works fine.

Switching to chat-format data to enable masking fixed two other things I had been
carrying without noticing: Qwen3's chat template emits an empty
`<think></think>` block, which is the *actual* mechanism for non-thinking mode;
I had been asking for it in English in the system prompt, which is not the same
thing. It also removed a stray leading space before every tool call that made the
training target tokenize differently from anything an inference path would
produce.

## Act 5: the thing that actually mattered

Here is the result I should have led with, and it is the least exciting one.

I had been quoting 90.7%. To check reproducibility I retrained with three
different random seeds, changing nothing else:

| seed | exact match |
|---|---|
| 0 | 76.7% |
| 1 | 81.4% |
| 2 | **90.7%** |
| 3 | 88.4% |

**Mean 84.3%, standard deviation 6.4 points, range 14 points.**

90.7% was not the result. It was the best of four draws. At this dataset size a
single run tells you almost nothing, and every comparison I had made (including
one where I concluded a data fix had caused a 14-point regression) was inside
the noise. That "regression" was seed 0 landing at the bottom of the
distribution. I nearly reverted a correct fix because of it.

After the config fixes (prompt masking, all 28 layers instead of 16, rank 16,
cosine schedule with warmup), across three seeds:

**Mean 88.4%, standard deviation 4.0.** Better mean, tighter spread, and a
validation signal that now points the right way.

## An interlude on measurement

Small-data fine-tuning has terrible measurement ergonomics. Nearly every mistake
in this project was a *measurement* mistake, not a modelling one:

1. A test set that wasn't held out
2. A metric that scored the easy half of the task
3. Labels that were confidently wrong
4. A test that asked for things never taught
5. A loss dominated by a constant
6. Single-run numbers with 14 points of spread
7. Gold labels in a stale format, scoring correct answers as wrong

The model was never the hard part. None of these were visible from the loss
curve. The first six all made the number look *better* than reality, which is why
they all survived. The seventh made it look catastrophically worse, and I nearly
abandoned a good idea because of it.

## Act 6: the part hyperparameters couldn't fix

The config work moved 84% to 88% and stalled. Every remaining failure was sibling
confusion between near-identical codes: `CUUR0000SAF11` (food at home) vs
`CUUR0000SAF1` (food), `CUUR0000SAM` (medical care) vs `CUUR0000SEMD` (hospital
services). Those aren't bugs. That's what memorizing a codebook into weights looks
like at the margin, and no learning rate fixes it.

I was training a 1.7B model to recall 13-character opaque strings where a
one-character slip silently fetches a different economic series. Lookup tables
don't belong in weights. So before writing that as an opinion, I measured it.

### Measuring the alternative

First a correction to my own framing. I had been saying the catalog has 8,103
series, so memorization covers 0.4% of it. That number is misleading. The catalog
is really **400 distinct items repeated across ~20 area and seasonal-adjustment
combinations**. "Housing" matches 120 titles because the same concept appears for
US city average, Northeast, New England, Chicago, seasonally adjusted and not.

Constrained to the namespace every one of my seeds actually uses (US city
average, not seasonally adjusted): the corpus is **400 rows, and the item name is
a unique key**. That reframes the problem: not 8,103 codes to memorize, but 400
well-named items to *look up*. Still 11x more than the 35 I could afford to
teach, but a completely different kind of problem.

So I built a ~30-line BM25 index over those 400 item names and measured recall of
the correct series ID on the same 43 held-out questions:

| query | recall@1 | recall@5 |
|---|---|---|
| oracle (the gold item's own name) | **100%** | 100% |
| **the raw user question, no model at all** | **84.4%** | **93.8%** |

The first row says the retriever has no ceiling problem: given a decent query it
is perfect. The second row is the uncomfortable one. Throwing the user's raw
question at BM25 (no model, no training, no GPU, no adapter) retrieves the
correct series 84.4% of the time, against 88.4% for the fine-tuned model. Those
are within noise of each other.

A week of distillation is currently tied with a text search over 400 strings.

And the two questions raw BM25 misses are precisely the two my fine-tuned model
gets wrong:

```
want  Medical care                          from "Show me healthcare CPI..."
want  Owners' equivalent rent of primary…   from "What did OER do between…"
```

Both are vocabulary gaps: "healthcare" and "OER" don't appear in the official
item names. So the obvious move is to have the model rewrite the user's words
into catalog vocabulary and let retrieval do the lookup. Much easier than
emitting a 13-character code from memory, and it generalizes to all 400 items.

I tested that too, and it does not work yet:

| query source | recall@1 | recall@5 |
|---|---|---|
| oracle (gold item name) | 100% | 100% |
| **raw question, no model at all** | **84.4%** | **93.8%** |
| base Qwen3-1.7B rewrites the question | 75.0% | 84.4% |
| **my fine-tuned model rewrites it** | **62.5%** | 71.9% |

Every model in the chain makes retrieval *worse*. The base model degrades good
queries. Asked about "tuition, other school fees, and childcare", which is
verbatim the official item name, it helpfully rewrites it to "Education
expenses". Asked about OER it expands the acronym to "Educational Resources
(OER)", which is a real term from a different field entirely.

My fine-tuned model is worse still, and the reason is the interesting part.
Asked for a search phrase, it emits series IDs:

```
"What did education and communication prices do…"  →  CUUR0000SAEDECU01
"Show me tuition and childcare CPI…"               →  CUUR0000SEEB
"Show me healthcare CPI…"                          →  search_phrase:CUUR0000SEMD Healthcare
```

It has specialized so hard on emitting codes that it can no longer paraphrase.
Note the second line: `CUUR0000SEEB` is the *correct answer*. The model knows the
mapping. It just can't express it in a form the retriever can use, because the
index is over item names and it only speaks in IDs.

That is a useful negative result. It means retrieval cannot be bolted onto this
adapter: the memorization fine-tune destroyed the exact capability retrieval
depends on. The two-step system has to be trained from base, on traces that
include the search step, and Anthropic's caution about small models and query
formulation is now something I've measured on my own data rather than quoted.

### Act 7: the fix was the output format

The agent ecosystem has converged on retrieval for a structurally identical
problem one level up. Hermes Agent, mcp-sieve, and various tool-router plugins
all replace "load every tool schema" with a `search` → `describe` → `call`
bridge. Anthropic's MCP evaluations show accuracy *improving* when tools are
deferred rather than preloaded (49% → 74% on Opus 4) because large catalogs
cause decision paralysis.

Their bottleneck is too many tools. Mine was 400 values in one argument slot. So
I was about to build the two-step system, and then noticed I could get most of
the benefit by changing one thing: **what the model is asked to emit.**

```
get_series(series_id="CUUR0000SAF11")   →   get_series(item="Food at home")
```

The model names the item; forty lines of code resolve the name to an ID. This
works only because of the 400-item finding above: item name is a *unique key*
in that namespace, so the resolution is deterministic, not a guess.

Why it helps is the same reason the failures were what they were. `SAF11` vs
`SAF1` is a one-character discrimination. *"Food at home"* vs *"Food"* is a
semantic one, which is the kind of distinction a language model is actually built
to make. And a name fails *loudly*: `resolve_item` returns `None` for something
it can't place, where one wrong character in a code silently fetches a different
series and returns plausible numbers.

Five seeds, same 43 held-out phrasings:

| target format | exact | sd |
|---|---|---|
| `get_series(series_id="CUUR0000SAF11")` | 88.8% | 3.0 |
| `get_series(item="Food at home")` | 92.1% | 1.3 |
| **+ hierarchy-aware resolver** | **94.4%** | **1.3** |

**+5.6 points, t = 3.8, p ≈ 0.005**, and seed variance more than halved.

{% include bls-progression.html %}

The hierarchy rule is small: when a bare term is a whole-word prefix of several
catalog entries, prefer the general one: "tobacco prices" means *Tobacco and
smoking products*, not *Tobacco products other than cigarettes*.

The only remaining failures at this point are "healthcare" → *Medical care* and
"OER" → *Owners' equivalent rent*. Both are vocabulary gaps the resolver cannot
bridge without knowing which official term the user meant. The alias table was
the obvious fix, and the obvious problem: both failures are visible from
reading the test set.

**And the first run of this experiment reported 23.3%.** A catastrophic
regression that would have killed the idea. It was a scoring bug: the held-out
file still stored raw `series_id` while the model now emitted item names, so
every *correct* answer scored wrong. I caught it because 10 of 43 passed, and
exactly 10 test items have no series ID. Gold is now rendered by the same
function that builds training targets, so the two cannot drift again.

That's the second phantom regression in this project. Both times the number said
"your change broke it" and both times the number was the broken thing.

A note on the "100% accuracy" claims circulating in the tool-router ecosystem: I
read them. One is 10 out of 10 test cases. Another is 100% on synthetic
validation data generated the same way as its training data. Both are the same
species of number as my original 93%. Copy the architecture; don't quote the
benchmarks.

### Act 8: the alias table

The two failures were "healthcare" → *Medical care* and "OER" → *Owners'
equivalent rent*. I said I was leaving them in. Then I shipped the alias table.

The number, same five adapters, model untouched:

| | exact | sd |
|---|---|---|
| fine-tuned, item names | 94.4% | 1.3 |
| + resolver alias table | 96.7% | 1.3 |

The accounting matters. There are roughly 50 aliases in the table: groceries,
gas, core CPI, airfare, public transport. Against this eval they move nothing;
its 43 questions don't use that vocabulary. The entire +2.3pp is `healthcare`.
That alias was added knowing it was one of my two held-out failures.

Which makes it contaminated by the definition this post uses. I reported 94.4%
as the model's number and put the alias row in a footnote.

The tension is that `healthcare` is simultaneously the most obvious synonym a
real user types and one of my two known test failures. Omitting it from a
user-facing resolver to protect a benchmark would be absurd. So no option was
both maximally useful and maximally clean.

What that means more broadly: once you've read your test failures, you can't
un-read them. The honest options narrow permanently. The stricter path was to
build the alias table from general domain vocabulary, validate it on the val
split, and leave test untouched. I didn't do that. I reported both numbers and
disclosed which one was contaminated. Honest, but not maximally rigorous, and
the distinction is real.

OER is still failing. The model emits `item="Education and communication"`, a
hallucination the resolver cannot repair. That is a model error, not a vocabulary
gap, and an alias does nothing for it.

One item remains genuinely open: enumerate all 400 names in context and let the
model select from a visible list rather than recall from memory. At 2,274 tokens
it fits, and it removes the lookup problem entirely for anything the model can
phrase correctly.

## What I would do differently

**Check your generation ratio before anything else.** If completions are short
relative to prompts, mask the prompt. Otherwise your loss is mostly measuring
whether the model can copy a constant, and any conclusion you draw from that
curve is suspect.

**Report a distribution, not a run.** Three seeds minimum, five if the difference
matters. At n=43 with 6 points of seed variance, a single number is close to
meaningless. If you only rerun the results you dislike, you will systematically
publish your luckiest draws.

**Write assertions, not intentions.** "The splits are disjoint" is a claim. A
build that fails when they aren't is a guarantee. Mine now refuses to run if any
series ID is absent from the catalog, if any example appears twice, if any
concept is missing from training, or if any label mentions dates its question
doesn't.

**Split by what the model must actually generalize over.** For a lookup task,
hold out phrasings. Holding out concepts measures the impossible.

**Ask what you're making the model say, not just how you're training it.** The
single largest improvement in this project (+5.6 points, variance halved) came
from changing the output format from an opaque ID to a name, not from any
hyperparameter. Errors that are one character apart are hard for a language
model; errors that are semantically apart are easy. And a name can fail loudly
where a code fails silently.

**Measure the dumb baseline first.** Thirty lines of BM25 over 400 strings scores
84.4% here. I was five days in before I knew that, and for all five of those days
I was comparing my results against nothing. If the baseline had come out ahead,
the right answer would have been to delete the model.

**Be most suspicious of results you like.** Six of the seven bugs above inflated
my numbers. That is not coincidence: unpleasant surprises get debugged, pleasant
ones get published.

**Once you've read your test failures, you can't un-read them.** The honest
options narrow. If you're building something that touches items you know failed,
validate it on val, not test. I didn't; I reported both numbers and disclosed the
contamination. Honest, not maximally rigorous. The difference matters.

## Results

43 held-out phrasings, none seen in training, every referenced concept present.
Mean over five training seeds, +/- one standard deviation across seeds:

| | tool | entity | exact |
|---|------|--------|-------|
| base Qwen3-1.7B | 72.1% | 9.3% | 9.3% |
| fine-tuned, emitting series IDs | 99.1% +/- 1.3 | 89.8% +/- 2.1 | 88.8% +/- 3.0 |
| **fine-tuned, emitting item names** | **99.1% +/- 1.3** | **94.4% +/- 1.3** | **94.4% +/- 1.3** |
| &nbsp;&nbsp;+ resolver alias table | 99.1% +/- 1.3 | 96.7% +/- 1.3 | 96.7% +/- 1.3 <sup>†</sup> |
| raw BM25 over 400 items, no model | -- | -- | 84.4% |

- **tool** — correct function chosen, out of six
- **entity** — plus the load-bearing argument (series ID / query / survey)
- **exact** — every argument identical

<sup>†</sup> Same five adapters, model untouched. The entire +2.3pp is the
`healthcare` alias, added knowing it was a held-out failure. See Act 8.

A 67MB adapter on an M4 Mini, for the 35 concepts it was taught. I want to
resist rounding that 99.1% to "100%"; four of five seeds hit 43/43 and the
temptation to quote the clean number is exactly the reflex this whole exercise
was about.

Worth stating the cost honestly: the config fix took the adapter from 19MB to
67MB, because it raised LoRA rank from 8 to 16 and adapted all 28 transformer
blocks instead of the last 16. So +4.5 points of exact match came with 3.5x the
adapter size. I did not sweep that tradeoff; rank 8 across all layers might get
most of the gain at half the size, and I have not checked.

The third row is there because a serious writeup should include the baseline
that embarrasses it.

One last operational note, learned the annoying way: I ran an evaluation
concurrently with a training job on the same GPU and got different numbers for
identical weights. Not non-determinism (two isolated runs are byte-identical),
but under memory pressure MLX returned different results rather than simply
running slower. I nearly wrote up a phantom regression from it. Score on a quiet
machine.

Code: [github.com/kovashikawa/bls_data](https://github.com/kovashikawa/bls_data).

## References

- Huerta-Enochian, M. & Ko, S. Y. (2024). "Instruction Fine-Tuning: Does Prompt
  Loss Matter?" EMNLP 2024. [arXiv:2401.13586](https://arxiv.org/abs/2401.13586)
- Guo, H., Dennis, S., Patil, R., & Shabahang, K. (2026). "When Mean CE Fails:
  Median CE Can Better Track Language Model Quality."
  [arXiv:2605.24667](https://arxiv.org/abs/2605.24667). Finds mean cross-entropy
  rising while held-out accuracy stays near peak, in Qwen2.5-1.5B SFT.
- Apicella, A., Isgrò, F., Pollastro, A., & Prevete, R. (2026). "Don't stop me
  now: Rethinking Validation Criteria for Model Parameter Selection."
  [arXiv:2602.22107](https://arxiv.org/abs/2602.22107). Finds early stopping on
  validation *accuracy* performs worst for neural classifiers, favouring
  loss-based criteria. Worth reading against this post: their objection is to
  early stopping specifically, and they find post-hoc selection across all epochs
  comparable to loss-based selection.
- Chatterjee, A., Renduchintala, H. S. V. N. S. K., Bhatia, S., & Chakraborty, T.
  (2025). "On the Effect of Instruction Tuning Loss on Generalization."
  [arXiv:2507.07817](https://arxiv.org/abs/2507.07817). Finds fully masking
  prompt tokens is rarely optimal either; a low-to-moderate prompt weight usually
  beats both extremes.
- Vaughn, D. (2024). "To Mask or Not to Mask: The Effect of Prompt Tokens on
  Instruction Tuning." Towards Data Science.
