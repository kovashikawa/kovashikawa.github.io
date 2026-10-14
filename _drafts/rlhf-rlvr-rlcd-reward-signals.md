---
layout: single
title: "RLHF vs RLVR vs RLCD: three reward signals compared"
excerpt: "RLHF rewards what people prefer, RLVR rewards what a checker accepts, and calibrated RL rewards being right about how likely you are to be right. The math behind each, with a rerunnable toy simulation."
description: "RLHF, RLVR and RLCD compared by their reward functions: preference loss, verifiable rewards with GRPO, and proper scoring rules, with worked numbers."
tags: [reinforcement-learning, llm, calibration, statistics, post-training, rlhf, rlvr, grpo]
categories: [ai]
author_profile: true
toc: true
toc_sticky: true
read_time: true
mathjax: true
last_modified_at: 2026-10-14
header:
  og_image: /assets/images/og/rlhf-rlvr-rlcd-reward-signals.jpg
---

RLHF (reinforcement learning from human feedback) rewards what raters prefer. RLVR (reinforcement learning with verifiable rewards) rewards what a checker accepts. RLCR (reinforcement learning with calibration rewards) and TypeSafe's RLCD (reinforcement learning for calibrated decisions) also reward stating probabilities that turn out to be right. This post compares the three reward signals, with the math.

A language model's behavior after post-training is mostly decided by one function, the reward. The optimizers differ in the details, but what the model gets better at, what it quietly gets worse at, and where you can trust it all trace back to what counts as a good answer.

One way to sort the three is by who consumes the output. RLHF targets a person reading a reply, so the thing to optimize is whether they approve. RLVR targets a checker, so it is whether the answer is correct. Calibrated training targets software that will act on the output, so it is whether the number attached to the answer can be trusted. The reward changes because the consumer does.

I got here from [a video on Jev](https://www.youtube.com/watch?v=YGgNBcIgI4s), a new model from TypeSafe that returns probabilities instead of text. TypeSafe says it is trained with RLCD and contrasts it with RLHF and RLVR in its [launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev). In the sources I read (mid-October 2026), TypeSafe has not disclosed the reward or the training algorithm, and Latent Space describes it as [a novel, unpublished technique](https://www.latent.space/p/jev). So this post does two things. It lays out RLHF and RLVR from their papers, and it uses a published relative of RLCD, RLCR from MIT, to show the math that calibrated training runs on. Where I am inferring about RLCD, I say so. Since Jev launched, Cloudflare, the Laya project and several independent authors have each described a recipe under the name RLCD, and the RLCD section compares them and the independent tests of Jev.

## RLHF vs RLVR vs RLCR: a shared objective, different rewards

The published methods can be read through one lens. Each maximizes expected reward while staying near a reference model:

$$
\max_{\theta}\; \mathbb{E}_{x\sim D,\; y\sim \pi_\theta(\cdot\mid x)}\big[\,R(x,y)\,\big] \;-\; \beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi_{\text{ref}}(\cdot\mid x)\big)
$$

This is a conceptual lens, not a description of the algorithms. InstructGPT and Tülu 3 both use PPO with a learned value model. GRPO, covered below, drops the critic, clips a per-token probability ratio, and adds KL as a separate penalty. Many open GRPO recipes drop the KL term entirely. TRL's trainer now defaults to $\beta=0$ and cites Open-Reasoner-Zero, Dr. GRPO and DAPO as showing the term is not essential ([TRL docs](https://huggingface.co/docs/trl/en/grpo_trainer)). Whether RLCD has a reference policy and a KL term at all is unknown, because Jev does not generate text. What the methods do share is that everything interesting is in $R$.

| | RLHF | RLVR | RLCR (published) | RLCD (TypeSafe) |
|---|---|---|---|---|
| $R(x,y)$ comes from | A reward model fit to human comparisons | A deterministic checker | A checker plus a proper scoring rule on stated confidence | Not published |
| Value | Unbounded scalar $r_\phi(x,y)$ | $\{0, \alpha\}$ | Bounded, $\mathbb{1}-(q-\mathbb{1})^2$ | Not published |
| Rewards | What raters prefer | What the verifier accepts | Being right and knowing how likely you are to be right | Per TypeSafe, probabilities that turn out to be right |
| Typical home | Chat assistants | Math, code, constraint following | Reasoning QA where confidence drives a decision | Decisions that software acts on |

## RLHF: preferences through a reward model

RLHF has two learned pieces. First a reward model $r_\theta$ is fit to human rankings. InstructGPT has labelers rank $K$ responses per prompt and trains on all $\binom{K}{2}$ pairs with a pairwise logistic loss ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)):

$$
\mathcal{L}_{\text{RM}}(\theta) = -\frac{1}{\binom{K}{2}}\;\mathbb{E}_{(x,y_w,y_l)\sim D}\Big[\log \sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big)\Big]
$$

This is a Bradley-Terry model. The reward gap is the log odds that a labeler prefers $y_w$ over $y_l$, so a gap of 0 is a coin flip, a gap of 1 is 73%, 2 is 88%, and 4 is 98%. Only differences are identified. Adding a constant to every reward changes nothing, which is why InstructGPT normalizes the model so labeler demonstrations average 0.

Then the policy is trained against $r_\theta$ with PPO. InstructGPT's objective adds a per-token KL penalty to the SFT model and an optional pretraining term (the PPO-ptx variant, an InstructGPT addition that many open recipes drop):

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y)\sim \pi_\phi^{RL}}\Big[r_\theta(x,y)-\beta\log\frac{\pi_\phi^{RL}(y\mid x)}{\pi^{SFT}(y\mid x)}\Big]+\gamma\,\mathbb{E}_{x\sim D_{\text{pretrain}}}\big[\log \pi_\phi^{RL}(x)\big]
$$

If you ignore optimization error and maximize the KL-regularized objective exactly, the optimal policy has a closed form (Eq. 4 of [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)):

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(r(x,y)/\beta\big)
$$

where $Z(x)$ normalizes. The reference model is tilted toward high-reward outputs, and $\beta$ sets how hard.

That closed form is also what DPO builds on. Solve it for $r$, substitute into the Bradley-Terry model, and the partition function cancels, leaving a loss on the policy itself:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim D}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big]
$$

There is no reward model and no sampling during training, and it runs offline on preference pairs. This matters when reading the literature, because Tülu 3's preference stage is DPO and its RLVR stage runs afterward. In practice "RLHF" covers PPO against a reward model and offline variants like DPO. This post follows the PPO form because its reward is explicit.

The weakness is the same in both. The target is $r_\theta$, a learned proxy for what raters like, and nothing in it ties the stated confidence to the probability of being right. The proxy can also be over-optimized. [Gao et al. (2022)](https://arxiv.org/abs/2210.10760) measured this in a synthetic setup with a fixed gold reward model standing in for humans, and found that pushing too hard on the proxy can hurt ground truth performance, as Goodhart's law predicts.

Approval has other documented costs. [Sharma et al. (2023)](https://arxiv.org/abs/2310.13548) found sycophancy across five RLHF-trained assistants. They also found that humans and preference models sometimes prefer convincing sycophantic answers to correct ones, and that on the hardest misconceptions the Claude 2 preference model chose the sycophantic response almost half the time (45%). They note that pretraining and SFT likely contribute as well. [Kirk et al. (2023)](https://arxiv.org/abs/2310.06452) found that RLHF generalises better than SFT but substantially reduces output diversity. Neither paper measures calibration.

Approval is a fair target for an assistant, where a person reads the reply and decides what to do with it. It is a poor target when software consumes the output. Approval is not correctness, and nothing in it tells a downstream system when to doubt the answer. The model has no reason to be indifferent to how it comes across, because that is exactly what it was paid for.

The evidence on calibration is consistent with that. OpenAI reports that the pre-trained GPT-4 is highly calibrated on a subset of MMLU, and that [after post-training the calibration is reduced](https://arxiv.org/pdf/2303.08774v6). That is a statement about token probabilities and about the whole post-training process, not RLHF alone. A later study found that for RLHF-tuned models, [verbalized confidences are typically better calibrated than their token probabilities](https://aclanthology.org/2023.emnlp-main.330/), often cutting expected calibration error by a relative 50%.

## RLVR: verifiable rewards replace the reward model

RLVR removes the learned proxy. Tülu 3 defines the reward as a verifier's verdict ([Lambert et al., 2024](https://arxiv.org/html/2411.15124v3)):

$$
v(x,y)=\begin{cases}\alpha & \text{if the verifier accepts } y\\ 0 & \text{otherwise}\end{cases}
$$

with $\alpha=10$ set from pilot runs and not tuned further. The training objective is the same KL-regularized form with $v$ in place of $r_\theta$. It only works where a program can decide correctness, which in Tülu 3 means math answers and checkable instruction constraints. In their 8B model, adding RLVR on top of the DPO checkpoint moved GSM8K from 84.3 to 87.6 and MATH from 42.0 to 43.7, so the gains are real but targeted.

### GRPO and the zero-gradient problem

Reasoning models are commonly trained with group relative policy optimization (GRPO), which replaces PPO's value network with a group baseline. For each prompt it samples $G$ completions, scores them, and standardizes within the group ([Shao et al., 2024](https://arxiv.org/abs/2402.03300), section 4.1.2):

$$
A_i=\frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}
$$

The full objective also has a clipped per-token probability ratio and a KL penalty. I show only the advantage. Because of the standardization, the scale of the reward drops out, so $\alpha$ matters for PPO and not for GRPO.

With a binary reward the standardization has a statistical consequence. If a prompt has true success probability $p$, the chance that all $G$ samples get the same reward, and therefore all advantages are zero, is

$$
P(\text{degenerate}) = p^{G}+(1-p)^{G}
$$

I computed it for a few cases.

| $p$ | $G=4$ | $G=8$ | $G=16$ |
|---|---|---|---|
| 0.05 | 0.815 | 0.663 | 0.440 |
| 0.20 | 0.411 | 0.168 | 0.028 |
| 0.50 | 0.125 | 0.008 | 0.000 |
| 0.95 | 0.815 | 0.663 | 0.440 |

Prompts the model almost always solves or almost never solves contribute nothing. [DAPO](https://arxiv.org/abs/2503.14476) responds by over-sampling and filtering out prompts whose accuracy is 0 or 1 ("dynamic sampling"), and reports reaching the same performance faster. Standardization has a second effect. With one success in 16, the lone success gets an advantage of about +3.9 and each failure about -0.26, so rare wins are amplified. For accuracy that concentration on the edge of the model's ability is the point.

### What RLVR does not reward

The reward is the same whether the model was sure or guessing. In the notation of the [RLCR paper](https://arxiv.org/abs/2507.16806), $R_{\text{correctness}}=\mathbb{1}_{y\equiv y^*}$, and the authors point out that it rewards a confident correct answer and a lucky guess identically, and penalizes an abstention and a wrong answer identically. Any confidence the model states is unconstrained by the reward. They cite prior work showing models drift toward overconfidence after RL, and note that reasoning models in particular show worse calibration and more hallucination than their base models.

This is the same incentive [Kalai et al. (2025)](https://arxiv.org/abs/2509.04664) identify in evaluation. Most benchmarks grade answers as right or wrong, so guessing beats abstaining, and they argue that is one reason models hallucinate. A binary training reward has the same shape.

Other papers report the same failure. An [ACL Findings 2026 study](https://aclanthology.org/2026.findings-acl.610.pdf) finds that RLVR produces extremely overconfident models while SFT stays better calibrated, and [DCPO](https://arxiv.org/abs/2603.09117) describes calibration degeneration under RLVR. I have read those two abstracts, not the papers.

## What calibration means, as a statistic

A predictor outputs an answer $\hat Y$ and a confidence $\hat P$. Perfect calibration is ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)):

$$
\mathbb{P}\big(\hat Y=Y\mid \hat P=p\big)=p\qquad\forall\, p\in[0,1]
$$

Of all the predictions made at 0.8, 80% should be right. You cannot test that for a continuous $\hat P$, so ECE (expected calibration error) bins predictions into $M$ intervals and takes the weighted gap between accuracy and confidence:

$$
\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}\,\Big|\operatorname{acc}(B_m)-\operatorname{conf}(B_m)\Big|
$$

ECE is a diagnostic, and it has a blind spot. A model that always says "62%" on a task it gets right 62% of the time has near-zero ECE and is useless, because the confidence carries no information about which answers are right. You want a measure that rewards both calibration and sharpness. That is what a proper scoring rule does.

A scoring rule $S(q,a)$ is proper if its expected value is optimized by reporting the true probability. For a binary outcome $a\sim\text{Bernoulli}(p)$, the Brier score $S(q,a)=(a-q)^2$ gives

$$
\mathbb{E}\big[(a-q)^2\big]=p(1-q)^2+(1-p)q^2,\qquad \frac{\partial}{\partial q}=2(q-p)
$$

so the minimum is at $q=p$. Log loss is also proper. So you can train a stated probability to be honest by making the reward a proper scoring rule, which is the move behind everything that follows.

### Three things called confidence

Papers use "confidence" for three different objects. A token probability is read off the model's output distribution, and the GPT-4 report measures this. A verbalized confidence is text the model writes ("I am 80% sure"), which Tian et al. and RLCR use. Because it is generated, training shapes it directly and can distort it. A typed-head probability is a distribution over a fixed set of options returned by a model that writes no text, which is the Jev interface. They differ in what is scored, what can be gamed, and how they behave under distribution shift, so a calibration result for one does not carry over to another.

## RLCR: correctness plus a Brier calibration term

RLCR trains a model to output an answer $y$ and a verbalized confidence $q$ after reasoning, with this reward ([Damani et al., 2025](https://arxiv.org/abs/2507.16806)):

$$
R_{\text{RLCR}}(y,q,y^*)=\mathbb{1}_{y\equiv y^*}-\big(q-\mathbb{1}_{y\equiv y^*}\big)^2
$$

Suppose the model has an answer that is right with probability $p$ and reports $q$. The expected reward is

$$
\mathbb{E}[R]=p-\Big(p(1-q)^2+(1-p)q^2\Big)
$$

I evaluated it for $p=0.7$.

| Reported $q$ | RLVR reward | RLCR reward |
|---|---|---|
| 0.0 | 0.700 | 0.000 |
| 0.3 | 0.700 | 0.330 |
| 0.5 | 0.700 | 0.450 |
| 0.7 | 0.700 | **0.490** |
| 0.9 | 0.700 | 0.450 |
| 1.0 | 0.700 | 0.400 |

The binary reward is flat in $q$. The RLCR reward peaks at $q=p$. At the optimum the expected reward is exactly $p^2$, which is increasing in $p$, so the model is still pushed toward the answer most likely to be correct. Between an answer with $p=0.6$ and one with $p=0.4$, each reported honestly, the expected rewards are 0.36 and 0.16.

The paper proves this as Theorem 1: the expected reward is maximized at $q=p_y$ (the calibration incentive), and among calibrated predictions it is maximized by the answer with the largest $p_y$ (the correctness incentive). It also says the result does not hold for an arbitrary proper scoring rule. The general form is $R=\lambda c - S(q,c)$, with $c$ the correctness indicator, and the theorem needs a bounded rule that satisfies $S(p,1)-S(p,0)\le\lambda$ for some finite $\lambda$. Brier passes, with $S(p,1)-S(p,0)=1-2p\le 1$. Log loss fails, because $S(p,1)-S(p,0)=\log\frac{1-p}{p}$ grows without bound as $p\to 0$.

The failure is about which answer the model prefers, and a small example shows it. With $\lambda=1$ and log loss, an answer that is right with probability $p$, reported honestly, earns $p-H(p)$ in expectation, where $H$ is the binary entropy in nats. At $p=0.5$ that is $-0.19$, and at $p\approx 0$ it is about 0. A model could do better by choosing an answer it is nearly sure is wrong and saying so. Brier gives $p^2$ instead, which never rewards that. So a log score is not unusable as an RL reward. [Rewarding Doubt](https://arxiv.org/abs/2503.02623), a different ICLR 2026 paper, uses a logarithmic scoring reward for verbalized confidence on factual questions and reports substantially improved calibration. RLCR's point concerns putting it in the same reward as correctness.

Reported results, from the paper's own abstract and introduction. On in-domain tasks RLCR matches RLVR accuracy while cutting ECE from 0.37 to 0.03 on HotpotQA and from 0.26 to 0.10 on a collection of math datasets. Out of domain, RLVR made calibration worse and RLCR improved it. The authors are explicit that out-of-domain calibration error is still high in absolute terms, and that models can still assign high confidence to contradictory answers. Their main comparison is against RLVR. I did not check how RLCR compares with simply recalibrating an RLVR model's confidence after training.

Group-relative training also gets more signal. In an all-correct group of 8 where the model states different confidences, binary advantages are all zero. Under RLCR the rewards differ, so the advantages are nonzero and the model is pushed toward higher confidence when every sample is right. These advantages use the population standard deviation. The sample standard deviation shrinks every one of them by about 6%.

```
confidences  .99  .90  .80  .70  .95  .60  .85  .50
rlcr reward  1.00 .99  .96  .91  .997 .84  .978 .75
advantage    +.85 +.73 +.38 -.21 +.82 -1.04 +.58 -2.11
```

The rewards here span only 0.75 to 1.00, and standardization stretches that to advantages between +0.85 and -2.11. The nudge toward higher confidence is right on average over many prompts, but any single group is noisy.

### A toy illustration of the metrics

To see how the metrics behave, I simulated 200,000 questions with true success probabilities drawn from Beta(2, 1.2) and realized outcomes sampled from them. The reported confidence is $q=\sigma\big(k\cdot\operatorname{logit}(p)\big)$. This shows what the metrics measure. It does not show that training finds the calibrated model, and it is not a measurement of any real model. The confidence is built from the true $p$, so $k=1$ is calibrated by construction. The other values of $k$ rescale it, with $k>1$ sharpening and $k<1$ flattening.

| $k$ | Accuracy | ECE | Brier | Log loss | RLVR reward | RLCR reward |
|---|---|---|---|---|---|---|
| 0.5 | 0.624 | 0.091 | 0.189 | 0.562 | 0.624 | 0.435 |
| 1.0 | 0.624 | 0.002 | 0.179 | 0.531 | 0.624 | **0.445** |
| 1.5 | 0.624 | 0.060 | 0.183 | 0.551 | 0.624 | 0.441 |
| 2.0 | 0.624 | 0.100 | 0.191 | 0.599 | 0.624 | 0.433 |
| 3.0 | 0.624 | 0.148 | 0.205 | 0.735 | 0.624 | 0.419 |
| 5.0 | 0.624 | 0.194 | 0.225 | 1.067 | 0.624 | 0.400 |

Accuracy never moves, so RLVR's reward cannot tell any of these six models apart. The RLCR reward is accuracy minus Brier here, so those two columns carry one signal, and ECE and log loss rank the models the same way. A few details are easy to misread:

- ECE at $k=1$ is zero in expectation. The 0.002 is finite-sample binning noise, and it ranged from 0.002 to 0.003 across 30 seeds.
- $k>1$ is overconfident only on average. For $p<0.5$ it pushes $q$ below $p$. Most of the Beta mass sits above 0.5, so the mean $q$ at $k=5$ is 0.69 against 0.62 accuracy.
- The log loss of 1.067 at $k=5$ is inflated partly by clipping $q$ at $10^{-6}$. That is a separate issue from the incentive problem above.

The baseline this table ignores is post-hoc recalibration. Because $k$ is a temperature ($q=\sigma(\operatorname{logit}(p)/T)$ with $T=1/k$), a single parameter fit on held-out data, which is the temperature scaling Guo et al. describe, would remove every row of miscalibration here without touching the model's weights. Any monotone remap also leaves ranking, and so AUROC, unchanged. Training-time calibration earns its cost when the model's confidence is poorly ordered, which a remap cannot fix, or when confidence has to be produced alongside the reasoning. I would treat the head-to-head as an open question.

## RLCD and TypeSafe's Jev: what is public, what I infer

TypeSafe's [primer](https://docs.typesafe.ai/introduction/machine-learning-primer) describes RLCD in three points. The model does not generate text. It returns decisions and probabilities. Higher probability should correspond to a greater chance the answer is correct. Their docs define calibration as outcomes assigned 0.2 happening about 20% of the time, and say it describes groups of predictions, not any single answer. The docs also credit cofounder Diogo Almeida with co-inventing RLHF. He is an author of InstructGPT (Ouyang et al., 2022). He is not an author of Christiano et al. (2017), the paper that introduced deep RL from human preferences.

I found no TypeSafe publication of the reward itself, and three independent write-ups say the same. My reading, and it is only that, is that the output contract makes the problem simpler than RLCR's. RLCR has the model choose an answer and attach a confidence, so the reward must couple them, and a bounded rule is needed to stop the model trading accuracy for score. A model that emits the full distribution $q(\cdot\mid x)$ over a fixed answer set can be scored directly with a proper rule against the observed label, with no answer-versus-confidence trade, which is why a log score would not hit the problem above. For the log score,

$$
\mathbb{E}_{y\sim p}\big[\log q(y)\big]=-H(p)-\mathrm{KL}(p\,\Vert\,q)
$$

is maximized at $q=p$. If that is roughly the shape of RLCD, it sits close to classical probabilistic classification, and the new part is the interface. Natural-language state in, typed questions (Choice, Score, Noul) out, in a single query. TypeSafe's launch post describes parallel sampling, and I do not know whether the probabilities come from a softmax head or from sample frequencies, which changes what a scoring rule would be applied to. I do not know the reward, and I would not cite any of this as TypeSafe's method. Published work on adjacent systems narrows the options, and I collect it in the next subsection.

What is public is the interface, and it makes calibration usable. A Choice answer returns a probability per option, and TypeSafe reports a `confidence` derived from that distribution. For $n$ options with top probability $p_{\max}$ the [documented formula](https://docs.typesafe.ai/confidence) is

$$
\text{confidence}=\frac{p_{\max}-1/n}{1-1/n}
$$

and for a Noul with probability $p$ it is $\lvert 2p-1\rvert$. Only the top probability counts, so $(0.6,0.3,0.1)$ and $(0.6,0.2,0.2)$ both come out at 0.4. This is a rescaled margin over chance, not the probability of being right. A confidence of 0.9 means $p_{\max}=0.95$ for two options and 0.933 for three, through $p_{\max}=\text{confidence}\cdot(1-1/n)+1/n$. Decision thresholds belong on the probabilities.

A typed output cannot be a malformed type, but it can still be a wrong valid value. [InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/) reports that objection from Hacker News against TypeSafe's "can't hallucinate" line, and quotes Armin Ronacher saying the design "delegates the hallucination problem a little bit to the user".

### What each approach hands back for the same question

Take one question, "is this message a refund request?", and ask it five ways. What you can read off the response differs a lot.

| Approach | What comes back | Is it the probability of being right? | Main catch |
|---|---|---|---|
| Ask the model to say how sure it is | Text such as "Yes, 85% sure" | It is a number the model wrote | Trained behavior shapes it, and post-training can push it toward overconfidence |
| OpenAI logprobs | `logprobs.content[]`, one entry per output token, each with `token`, `logprob` and a ranked `top_logprobs` list | It is the probability of a token, such as "Yes" against "No" if you force a one-word answer | OpenAI's own cookbook notes that it shows certainty in the classification, not whether the classification is correct |
| Gemini logprobs | `logprobsResult` with `chosenCandidates` and `topCandidates` per step, plus `avgLogprobs` | Same as above | Google's reference marks the `logprobs` parameters deprecated for Gemini 3.x |
| Claude (Anthropic Messages API) | No token probabilities that I could find | Not available | Use a verbalized score, a sampling-based method, or another provider |
| Cleanlab TLM | `{response, trustworthiness_score}` with a score from 0 to 1 | A score estimated from several LLM calls on top of a base model (the docs list consistency, self-evaluation and token probabilities) | Not deterministic, results are cached, and Cleanlab says to pick thresholds per application |
| TypeSafe Jev | One answer per question id. Choice returns `choice`, `probabilities` (one per option) and `confidence`. Noul returns one number from 0 to 1 | It is meant to be, per TypeSafe, and the probabilities are the primary output | Independent tests of the calibration are mixed (below), and wrong but valid answers remain possible |

I have not called any of these APIs for this post. The field names come from each provider's documentation. The Claude row rests on a gateway's documentation and a maintainer's reply on an SDK issue, not on Anthropic's API reference. Cleanlab also publishes benchmarks claiming its score beats logprobs and LLM-as-judge at catching wrong answers, and those are the vendor's numbers.

The practical difference is where the probability lives. Logprobs are a byproduct of a model trained to write text, so you get the probability of the next token, and you have to engineer the prompt so that one token carries the decision. A wrapper such as TLM estimates trust from the outside. Jev returns the probability as the answer itself.

### Clues from adjacent work

TypeSafe has not disclosed its RLCD recipe in the sources I read. Four earlier lines of work train a model to emit a probability under a proper scoring rule. The next two subsections cover newer decision models and independent RLCD implementations. Read together, they bound what TypeSafe's could look like.

| Work | What is trained | Objective | Output | Reported calibration |
|---|---|---|---|---|
| [Turtel et al., 2025](https://arxiv.org/abs/2505.17989) | 14B reasoning model, RL (GRPO variants, ReMax) | $R=-(\hat p-y)^2$, Brier on a resolved event | One probability for a binary question | Matches o1 on Brier (0.193), beats it on ECE (0.042, $p<0.001$); arXiv v2 abstract |
| [Generative Verifiers](https://arxiv.org/abs/2408.15240) | LLM, supervised next-token loss | $-\log p_\theta(\text{Yes}\mid x,y^+)$ and $-\log p_\theta(\text{No}\mid x,y^-)$ | $p_\theta(\text{Yes})$ as a one-token score | Not reported in the parts I read |
| [Kadavath et al., 2022](https://arxiv.org/abs/2207.05221) | LLM plus a value head ("P(IK)") | Binary classification of whether the model answers correctly | A probability from the head | Well calibrated on the training distribution, poorly on new tasks |
| [PFNs](https://arxiv.org/abs/2112.10510) and [TabPFN](https://www.nature.com/articles/s41586-024-08328-6) | Transformer, supervised on synthetic datasets | $\mathbb{E}\big[-\log q_\theta(y\mid x, D_{\text{train}})\big]$ | A class distribution in one forward pass | ECE 0.025 (the BNN-prior PFN) against 0.066 for XGBoost, on tabular subsets with 30 training samples |

Kadavath et al. also report that large pretrained models are well calibrated on lettered multiple-choice and true/false formats before any such training, which is the token-probability baseline from the GPT-4 report above. The accepted TMLR version of Turtel et al. reports different figures on a smaller hold-out set (Brier 0.190, ECE 0.062).

Three things follow. First, a bounded Brier reward works as an RL reward when the model is not choosing between answers. Turtel et al. found that standard GRPO pushed policies toward extreme overconfidence or collapse on noisy, delayed outcomes. They dropped the standard deviation division and used ReMax, which keeps gradients proportional to the Brier loss and matches the standardization caveat in the GRPO section. Second, when labels already exist, ordinary cross-entropy is a log score, so a typed probability gets a proper scoring rule from supervised training alone. GenRM and PFNs do exactly that. It makes it plausible that part of RLCD is not RL at all, though TypeSafe calls it RL and I do not know. Third, the calibration is conditional on the training distribution. Kadavath et al.'s P(IK) lost calibration on new tasks, and a PFN is calibrated with respect to its synthetic prior. That is the same warning as the base-rate paragraph below.

The retrofit route is also published. [Leng et al. (2025)](https://arxiv.org/abs/2410.09724) show that RLHF reward models are biased toward high verbalized confidence regardless of answer quality, and their PPO-M and PPO-C variants lower ECE without losing accuracy. That adds calibration to a preference reward. The four works above start from a proper scoring rule instead.

None of these four is an LLM that reads free-text state and returns several typed answers in one pass, and I did not see calibration under adversarial input reported in the parts I read. The decision models in the next subsection do the first part.

### Other decision models and what each one publishes

Jev is no longer alone. Since its launch, several groups have shipped a model with the same interface, a state plus typed questions in and probabilities out. They differ a lot in how much they say about training, and that is the useful part here, because it shows what the recipes behind the name contain. I read each vendor's own pages. Every benchmark number below is the vendor's, run on a suite the vendor chose.

| Model | Base and size | Documented training | Output | Where it leans |
|---|---|---|---|---|
| [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions) (`gpt-6-luna`) | Not stated | Nothing in the docs I read | `predicate` returns one probability. `choice` and `score` return `probabilities` as a list plus `confidence`. Examples show 0 output tokens. $0.10 per million input tokens through [LLM Gateway](https://docs.llmgateway.io/features/decisions) | Distribution and speed. It claims about 10 times faster than the Responses API, and gateways now copy its wire format |
| [QwenCloud `decision-model-preview`](https://docs.qwencloud.com/api-reference/decision-model-api) | Not stated | Nothing in the API reference I read | System One format at `/compatible-mode/v1/systemone`. No streaming, function calling or structured output | A hosted preview that copies Jev's interface, field names included |
| [Cloudflare Clef and Clef-flash](https://blog.cloudflare.com/clef-decision-models/) | Frozen Qwen3.8-27B and Qwen3.5-9B, a routing head and rank-256 LoRA adapters. Apache 2.0 weights | Label-smoothed cross-entropy plus a Brier loss. Then "RLCD" as a secondary stage with partial credit for adjacent ordinal choices, a reward for exactly right records and a reference penalty | Jev's System One format. Adds up to 4 images and a 64k context | Supervised proper loss first, RL as a refinement. Competes on latency and breadth |
| [Microsoft-Decision-1](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/) | Qwen3.5-9B, post-trained | "Publicly available datasets" plus synthetic data. No loss or RL stated | A calibrated probability per option, text only. $0.042 per million input tokens on LLM Gateway | Enterprise distribution through Foundry. Says it will rebase on other models |
| [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) | Frozen Qwen3.5-4B-Base, LoRA rank 16 and a pointer head. Apache 2.0 | Cross-entropy over the options, in four stages. No RL. One temperature, $T=2.41$, fitted by negative log-likelihood. The card says no Jev output was used | System One format | Supervised loss plus post-hoc calibration. The cheap, self-hosted end |
| [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large encoder, 421M, no text generation | RL with a group-mean baseline, noise on the logits, and a reward from strictly proper scoring rules (log, spherical, and a ranked score for ordinal questions), alongside soft cross-entropy against a teacher | System One format, self-hosted | The most literal reading of "RLCD is a proper scoring rule". Competes on size and latency |
| [jevper](https://github.com/zhulinchng/jevper) | Any OpenAI-compatible model | None, it is a wrapper | Softmax over the label logprobs of the first answer token where the provider returns them. Otherwise the model's self-reported probabilities, or one label | Decision-shaped output on a general LLM. Reasoning models return no logprobs, so it falls back to self-report |

Qwen is the backbone of the Microsoft, Cloudflare and Kev models, and of community checkpoints such as [Open-Jev-9B](https://huggingface.co/junetask/Open-Jev-9B), [RSI-Jev](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-2b) and [jev-qwen35-08b](https://huggingface.co/guoan1/jev-qwen35-08b). That list is a sample, not a census.

The interface is nearly the same everywhere, with a few renames. The table below is from LLM Gateway's documentation of how its two routes map onto each other.

| | Jev, Clef, Kev (System One) | OpenAI Decisions |
|---|---|---|
| Route | `/v1/systemone` | `/v1/decisions` |
| Evidence | `state` | `input` |
| Questions | A map keyed by question id | An array, with the id in `name` |
| Yes or no | `noul` question, `noul` answer | `predicate` question, `probability` answer |
| Options | `criteria` record | `choices: [{value, description}]` |
| Probabilities | A record keyed by option | An array of `{value, probability}` |

The recipes use three losses between them, with $K$ options, a reported distribution $q$ and an observed label $y$.

$$
\mathcal{L}_{\text{CE}}=-\sum_{k=1}^{K}\Big[(1-\varepsilon)\,\mathbb{1}[k=y]+\tfrac{\varepsilon}{K}\Big]\log q_k
\qquad
\mathcal{L}_{\text{Brier}}=\sum_{k=1}^{K}\big(q_k-\mathbb{1}[k=y]\big)^2
$$

Cross-entropy with label smoothing $\varepsilon$ is the log score with a softened target. The Brier loss is the quadratic score. Laya adds the spherical score, $S_{\text{sph}}(q,y)=q_y/\lVert q\rVert_2$, and for ordered levels the ranked probability score,

$$
\mathrm{RPS}(q,y)=\sum_{k=1}^{K-1}\Big(\sum_{j\le k}q_j-\mathbb{1}[y\le k]\Big)^2 .
$$

The RPS charges more for probability placed far from the true level than for probability placed next to it, which is what "partial credit to adjacent ordinal choices" would amount to in a reward. Gneiting and Raftery's [review of scoring rules](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) covers the propriety of the logarithmic, spherical and quadratic scores. The RPS is [Epstein's (1969)](https://journals.ametsoc.org/view/journals/apme/8/6/1520-0450_1969_008_0985_assfpf_2_0_co_2.xml) score for ranked categories, the sum of squared differences between cumulative forecast and cumulative observed probabilities. The last cumulative term is always zero, which is why the sum stops at $K-1$, and some authors divide by $K-1$.

The RL stage in these recipes deserves a closer look. Laya's card describes REINFORCE with a group-mean baseline,

$$
\nabla_\theta J \approx \mathbb{E}\Big[\big(R-\bar R\big)\,\nabla_\theta \log \pi_\theta\Big],
$$

with $R$ a scoring rule applied to the reported distribution. Here is my own inference. A proper score is a differentiable function of $q$, so the exact gradient is available without sampling. For the log score and softmax logits $z$ it is $\nabla_z \log q_y = e_y - q$, the same gradient as cross-entropy. REINFORCE then estimates something supervised training already computes, with more variance. A reason to pay for it is a reward that is not differentiable, and "fully precise record outputs" in Cloudflare's description is that kind of reward, an exact match across several fields. This is why I read the supervised stage as the load-bearing one and RL as the part that handles what a loss cannot express. An independent control experiment, below, finds a sampled proper reward no clearly better than the direct losses. It is a reading of other people's recipes, not a statement about TypeSafe's.

Three things follow.

1. **Supervised proper loss first.** Cloudflare uses cross-entropy and Brier before RL. Kev uses no RL at all and reaches usable probabilities with cross-entropy plus one temperature, the post-hoc fix from [Guo et al.](https://arxiv.org/abs/1706.04599). A fitted $T$ above 1 flattens a softmax, so I read that as the raw head having been overconfident.
2. **"RLCD" is now several recipes.** Cloudflare's rewards ordinal closeness, exact records and staying near a reference model. Laya's rewards a proper score on the whole distribution. The next subsection adds independent ones. None says it is TypeSafe's, and TypeSafe's is not public.
3. **The vendors' own calibration evidence is thin and mixed.** Laya's own card reports that its fine-tuned checkpoint is still overconfident, with ECE 0.213 against 0.144 for Jev, and it fine-tunes on the benchmark's training split. It also says the Jev figures are third-party and not measured on the same prompts. The card for its general checkpoint reports the reverse on a different suite, ECE 0.081 against 0.246 for Jev after temperature fitting, so the same project shows both directions. The Cloudflare tables I read report accuracy and latency but no calibration. In Cloudflare's own comparison Clef beats Jev on most rows, and Jev is ahead on When2Call, BRIGHT and agent-trace observability. Clef-flash falls well behind on CLINC150 (66.8 against 89.3 macro-F1). None of that says whose probabilities are better calibrated, which is the claim the whole category rests on. Independent tests of Jev are in the next subsection.

### Independent RLCD implementations and tests of Jev

Since the launch, authors with no tie to TypeSafe have published both implementations of RLCD and tests of Jev. I read the abstracts of all of them and, for OpenJev-RLCD, the algorithm section. I did not reproduce any result.

Three independent implementations, with the reward each one uses:

| Work | Model | Reward or objective | What they report |
|---|---|---|---|
| [OpenJev-RLCD](https://arxiv.org/abs/2609.38850) (Gao and Wang) | Qwen3-1.7B reasoning model | Each sampled rationale is scored by a strictly proper score of the answer distribution the model commits to afterwards. Stage 1 trains that readout with the proper score. Stage 2 reinforces the reasoning with the same score as reward, plus a KL anchor to the stage-1 model | Matches or beats SFT, RFT/STaR and GRPO (each temperature-scaled) on accuracy and beats them on selective prediction. On GSM8K answer verification, one query decides 81% of items at 5% error or less, against 19% for GRPO (HTML version) |
| [JevRL](https://jevrl.github.io/output/pdf/jevrl_paper.pdf) (Tegong, one-seed technical report) | Frozen Qwen2.5-0.5B-Instruct and a 142K-parameter typed head | Two sampled answers $A, B$, rewarded $\mathbb{1}[A=y]+\mathbb{1}[B=y]-\mathbb{1}[A=B]$ | On a synthetic set of 300 test items: accuracy 0.611, Brier 0.093, ECE 0.040. Cross-entropy and Brier heads trained on the same labels score 0.688 and 0.700 accuracy, Brier 0.058 and 0.038, ECE 0.054 and 0.060. A single-sample accuracy reward gives ECE 0.393 |
| [Jev Cookbook](https://datawhalechina.github.io/jev-cookbook/materials/02-nanojev/pages/docs/RLCD_EXPERIMENT/) (Datawhale) | A Qwen3-0.6B based checkpoint and a small MLP | A sampled paired proper reward against direct cross-entropy and Brier controls | "The evidence does not establish that the sampled method is better than direct controls." The paired reward has the lowest error in one run and direct Brier has lower out-of-distribution NLL |

Two of the three papers also say outright that TypeSafe has not disclosed its algorithm, and the cookbook calls its formula "an independent research candidate, not a recovered official recipe".

The two identities behind these recipes are short. Take $M$ sampled answer distributions $u_1,\dots,u_M$, their mean $\bar u$, and the reward $J(u)=-\lVert u-e_y\rVert^2$, where $e_y$ is the one-hot vector for the observed label. For two draws $A,B$ from a distribution $p$, JevRL's reward has expectation

$$
\mathbb{E}\big[\mathbb{1}[A=y]+\mathbb{1}[B=y]-\mathbb{1}[A=B]\big]=2p_y-\lVert p\rVert^2=1-\lVert p-e_y\rVert^2 ,
$$

which is one minus the Brier score. So the duplicate penalty is what turns a pair of accuracy rewards into a proper one. I checked the identity by simulation for $K=2,3,5$. OpenJev-RLCD's variance identity says that scoring the average of several sampled distributions pays a bonus for their spread,

$$
J(\bar u)=\frac1M\sum_{i=1}^{M}J(u_i)+\frac1M\sum_{i=1}^{M}\lVert u_i-\bar u\rVert^2 ,
$$

and when each answer is one-hot it becomes $J(\bar u)=2\,\mathrm{acc}-2+\big(1-\sum_k\bar u_k^2\big)$. That is the RLVR accuracy term plus a Gini diversity bonus whose coefficient is fixed at one, which is the paper's link between RLVR and calibrated rewards. Scoring each rationale's own distribution removes the bonus. I confirmed both forms numerically.

Tests of Jev itself, by authors outside TypeSafe, are mixed on calibration and not reassuring on attacks:

| Work | What was tested | What it found |
|---|---|---|
| [Just Ask Jev](https://arxiv.org/abs/2609.29429) (Guo et al.) | Jev as a zero-shot detector of alignment failures, 44 benchmarks | Median AUROC 0.886 with one generic question. Pooled ECE is 0.047, but the median per-benchmark ECE is 0.168 against 0.074 under perfect calibration. The error is a base-rate mismatch, not a ranking problem (median AUROC 0.905). Keeping the more confident half of decisions raises median accuracy from 0.793 to 0.933 |
| [Jev vs. LLMs as Rubric Judges](https://arxiv.org/abs/2609.29769) (Rao and Callison-Burch) | Jev against three flash-tier LLM judges on nine rubric panels | Accuracy differs significantly in 8 of 27 paired comparisons. The LLM judges cost 29 to 325 times as much. They repeat nearly all of Jev's most confident errors, so a cascade gains at most 1.5 points |
| [Same-Number Citation Swaps](https://arxiv.org/abs/2610.08675) (Xu et al.) | Jev as a verifier of cited roles in financial calculations | Wrong-role citations pass and valid alternative citations are withheld. A literal quotation check already catches all 27 unsupported-role traces |
| [Adversarial Cues](https://arxiv.org/abs/2610.11436) (Liu) | Jev as an answer judge, one colon added to a candidate | False acceptance rises from 1.0% to 26.0% with three labels and from 3.0% to 26.5% with the published four-label instruction |
| [The Option-Channel Attack](https://arxiv.org/abs/2610.12292) (Azizi et al.) | Seven open-weight typed decision models as agent guardrails, not Jev itself | Guardrail accuracy 36% to 72% against 50% chance. Six lines of irrelevant log text raise fail-open from 0% to 63%. A misleading option name raises it to 93% to 100% on four models. Escalating the least confident decisions does not help |

The calibration result is the useful one for anyone building on the probabilities. Pooled over many tasks the probabilities look calibrated, and within a single task they are off by a base rate. That is the base-rate paragraph earlier in the post, measured on Jev. The attack results are about the model class and the interface, not about the reward, but they bear on the same question, since a calibrated number on a manipulated input is still a wrong number.

### Where Jev seems to lean

Putting the interface and the adjacent work side by side, here is what points where. This is inference, since I found no public statement of the training recipe.

| Observation | Leans toward |
|---|---|
| The answer is a distribution over options or levels you declared, one per question id | The typed-head family: GenRM, P(IK), PFNs. Not a reasoning model that writes a number, and not a wrapper |
| TypeSafe's docs describe training decision models on calibrated probabilities instead of generated text | A probability that is trained for, not read off a text model afterward |
| `confidence` is computed from the returned distribution with a fixed formula | No separate uncertainty head. Everything downstream comes from the distribution |
| You can send many questions in one call, including speculative ones | A cheap per-question cost. Whether that is one forward pass or parallel sampling is unclear, since the launch post says sampling |
| TypeSafe calls the method RL, and its launch post contrasts it with RLHF and RLVR | Some RL stage, but nothing says its reward is a proper scoring rule |
| Independent implementations (OpenJev-RLCD, JevRL) read the name as a proper score on the reported distribution, with a calibrating stage before RL in OpenJev | A reading of the name by outsiders, not TypeSafe's recipe |
| The vendors that copy the interface describe a supervised stage with cross-entropy and Brier loss, with RL optional | A supervised proper loss probably does most of the work, and RL refines it. Inference from other people's recipes |

My read is that Jev is closest to a typed-head model trained against a proper score, possibly with an RL stage on top, and furthest from token logprobs or an external wrapper. Cloudflare, Laya and independent authors each use the name for their own recipes, and nothing I read says any of them matches TypeSafe's. Two public facts would change my read. A technical report with the loss would settle it directly. A comparison of calibration before and after the RL stage would show whether the RL step adds anything over supervised training on the same labels. In the meantime, TypeSafe's calibration claims have independent tests, and the pooled picture is mixed: calibrated when averaged over many tasks, off by a base rate within a task.

## How each one gets gamed

**RLHF.** The reward model is a proxy, and pushing the policy hard against it eventually costs real quality (Gao et al., above). A rater who cannot check a claim can still prefer the confident phrasing, so a proxy trained on those choices can reward sounding right.

**RLVR.** A verifier is only as good as its coverage. Typical failures are answers that match the checker's format without being right, special-casing the test cases, and false negatives that punish correct answers. It also says nothing about confidence.

**RLCR.** Properness is a statement about the optimum, not about optimization. A [paper on confidence rewards](https://arxiv.org/abs/2607.04332) describes confidence reward hacking, where a poorly designed scheme lets the model gain by answering wrongly so that it can be confident the answer is wrong, and reports that it occurs on practical datasets. Two more papers put the problem in the optimizer. DCPO reports a gradient conflict between maximizing accuracy and minimizing calibration error, and [CAPO](https://arxiv.org/abs/2604.12632) argues that GRPO's uncertainty-agnostic advantage degrades relative calibration. Again, I have read abstracts.

**RLCD.** Unknown for TypeSafe, because I found no published reward. Partly known for the others, since Cloudflare, Laya and independent authors describe recipes. Two caveats come from outside the reward. Calibration is an average-case property, not a guarantee against crafted input, and the independent attack studies above show typed decision models failing open under small edits to the input or to the option names. For guardrails and routing the state often contains untrusted text, so treat it as hostile and keep non-model checks on high-cost actions. And typed outputs do not prevent wrong valid values, as above.

## Where each one belongs

Calibration is what lets software choose a threshold with arithmetic. Say an automated action costs $c_{FP}$ when it fires wrongly and skipping a true positive costs $c_{FN}$. If $p$ is a calibrated probability, the expected cost of acting is $(1-p)\,c_{FP}$ and of not acting is $p\,c_{FN}$, so act when

$$
p>\frac{c_{FP}}{c_{FP}+c_{FN}}
$$

With illustrative numbers, a wrongly issued refund costing 50 and a missed one costing 5, the threshold is $50/55\approx0.91$. In the video that prompted this post, the example cutoff on the returned probability was 0.9, close to this.

Two assumptions sit under the rule. Correct actions and correct skips cost nothing, and the probability is calibrated on your own traffic. Add a human-review cost $c_h$ and a review band exists only when $c_h < c_{FP}c_{FN}/(c_{FP}+c_{FN})$, which is 4.55 here. At $c_h=2$ you review between $p=0.40$ and $p=0.96$ and act automatically only above $1-c_h/c_{FP}=0.96$. Thresholded decisions also depend on how well the score ranks right answers above wrong ones (AUROC, risk-coverage curves), not only on calibration.

Calibration is conditional on the task and the data distribution. A model calibrated on a set that is half positives can be miscalibrated when prevalence falls to 5%, and RLCR's own out-of-domain error stays high. Check reliability on a labeled held-out sample from your own traffic, and again after any model version change. InfoQ's summary of TypeSafe's docs advises pinning a version rather than using the moving alias. A threshold chosen on a miscalibrated score is a guess with extra steps.

| Method | Use it for | Evidence in this post | Watch for |
|---|---|---|---|
| RLHF | Open-ended assistants, tone, refusals, anything with no ground truth | InstructGPT's 1.3B model preferred over 175B GPT-3 by labelers | Reward model over-optimization, and calibration loss after post-training |
| RLVR | Math, code, formats, constraints a program can check | Tülu 3 gains on GSM8K, MATH, IFEval | Overconfidence, zero-gradient prompts, only works with a verifier |
| RLCR | Reasoning QA where the stated confidence drives a decision, plus test-time confidence-weighted voting | ECE 0.37 to 0.03 on HotpotQA | Out-of-domain calibration still imperfect, confidence reward hacking |
| RLCD | Routing, guardrails, classification, reranking, verification steps in a pipeline | TypeSafe's cookbooks, which are vendor demos | Not generative, average-case calibration, wrong valid values, jagged on counting, dates and large noisy state |

Two of those rows deserve a sentence each. RLCR's confidence is useful at inference time. The paper's confidence-weighted majority voting and max-confidence selection both improve accuracy and calibration as compute scales. For RLCD, TypeSafe's own [jaggedness page for jev-1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13) says it is not a calculator, reads literally, and degrades when the state is full of irrelevant detail. I have not tested Jev myself, and its cookbook numbers are the vendor's.

Calibrated confidence has uses beyond the four rows. These apply to calibrated confidence in general, not to any one method:

- **LLM-as-judge.** A calibrated judge score can be thresholded or averaged sensibly. An uncalibrated one cannot.
- **Selective prediction.** Answer or abstain, evaluated with risk-coverage curves and not ECE alone.
- **Test-time compute.** Spend more samples where confidence is low.
- **Tool-call gating in agents.** Allow, ask, or block based on a probability.
- **Retrieval-augmented answers.** Answer, or say the sources do not contain it.

## RLHF, RLVR and RLCD in short

The three differ most in who the output is for. RLHF optimizes a model of what people prefer, which suits an assistant. RLVR optimizes a checker's verdict and gets sharper accuracy with no pressure on honesty about uncertainty. Calibrated RL adds a proper scoring rule so that the stated probability is itself rewarded for being true. RLCR is RLVR plus a term, since its reward contains the binary correctness reward. RLCD is a different model class. TypeSafe's recipe is unpublished. The others that use the name, Cloudflare's, Laya's and OpenJev-RLCD, each pair a proper-score stage with an RL stage, in different orders. A stack that uses RLVR for what a program can check, RLHF for what it cannot, and calibrated outputs wherever another system will act on the number is my suggestion, not a tested recipe. The related work above reports conflicts between accuracy and calibration objectives, so I would expect to measure rather than assume that they compose.

## Appendix: notation

RLCD means reinforcement learning for calibrated decisions throughout. One symbol set is used across the post, with two exceptions noted at the end.

| Symbol | Meaning |
|---|---|
| $x$, $y$ | A prompt or state, and an output or observed label |
| $\pi_\theta$, $\pi_{\text{ref}}$ | The policy being trained and the frozen reference policy |
| $R(x,y)$ | The reward. $r_\theta$ is a learned reward model and $v$ a verifier's reward |
| $y_w$, $y_l$ | The preferred and the rejected response in a human comparison |
| $\sigma$ | The logistic sigmoid |
| $\beta$, $\gamma$ | The KL weight, and the weight on the pretraining term in InstructGPT |
| $\alpha$ | The verifier's reward value in Tülu 3 |
| $G$ | The number of completions sampled per prompt in GRPO |
| $p$, $q$ | The true probability of an outcome, and the probability a model reports |
| $\hat Y$, $\hat P$ | A predicted answer and its stated confidence |
| $K$ | A count, see below |
| $n$ | The number of options in the Jev `confidence` formula |
| $\varepsilon$ | The label-smoothing weight |
| $T$ | A softmax temperature |
| $\mathbb{1}[\cdot]$ | The indicator function |
| ECE, Brier, RPS | Expected calibration error, the quadratic score, the ranked probability score |

$K$ means two things. In the RLHF section it is the number of responses a labeler ranks, as in InstructGPT. From the calibration sections on it is the number of answer options. In the same way, $\theta$ parameterizes the reward model in the RLHF section, following InstructGPT, which uses $\phi$ for the policy, and it parameterizes the policy everywhere after that.

## Reproduce the numbers

The snippet is self-contained and reproduces every table in this post except the TypeSafe and Tülu figures, which come from their sources. It is about 40 lines of NumPy.

```python
import numpy as np
rng = np.random.default_rng(0)
sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda x: np.log(x / (1 - x))

# Bradley-Terry: P(preferred) for reward gaps 0, 1, 2, 4
print("BT", sig(np.array([0, 1, 2, 4])).round(3))

# P(all G samples get the same binary reward) at success probability pp
for G in (4, 8, 16):
    print("G", G, {pp: round(pp**G + (1 - pp)**G, 3) for pp in (0.05, 0.2, 0.5, 0.8, 0.95)})

# expected RLCR reward when the true success probability is 0.7
p0 = 0.7
for q0 in (0, .3, .5, .7, .9, 1):
    print("q", q0, round(p0 - (p0 * (1 - q0)**2 + (1 - p0) * q0**2), 3))

# GRPO advantages for an all-correct group (population std, ddof=0)
conf = np.array([.99, .90, .80, .70, .95, .60, .85, .50])
r = 1 - (conf - 1)**2
print("adv", ((r - r.mean()) / r.std()).round(2))

# toy simulation
n = 200_000
p = rng.beta(2, 1.2, size=n)                   # true success probability per question
y = (rng.random(n) < p).astype(float)          # realised correctness

def ece(q, y, M=15):
    b = np.minimum((q * M).astype(int), M - 1)
    return sum((b == m).mean() * abs(y[b == m].mean() - q[b == m].mean())
               for m in range(M) if (b == m).any())

print("k acc ece brier logloss rlcr")
for k in [0.5, 1, 1.5, 2, 3, 5]:
    q = np.clip(sig(k * logit(np.clip(p, 1e-6, 1 - 1e-6))), 1e-6, 1 - 1e-6)
    brier = ((q - y) ** 2).mean()
    logloss = -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
    r_rlcr = (y - (q - y) ** 2).mean()
    print(k, y.mean().round(3), ece(q, y).round(3), brier.round(3), logloss.round(3), r_rlcr.round(3))
```

## Sources

- Christiano et al., [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741) (2017)
- Ouyang et al., [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT; reward model loss, PPO-ptx objective)
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (closed-form optimal policy, DPO loss)
- Gao et al., [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- Lambert et al., [Tülu 3](https://arxiv.org/html/2411.15124v3) (RLVR, $\alpha=10$, benchmark gains)
- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300) (GRPO, advantage in section 4.1.2)
- Yu et al., [DAPO](https://arxiv.org/abs/2503.14476) (dynamic sampling)
- Damani et al., [Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty](https://arxiv.org/abs/2507.16806) (RLCR; ICLR 2026)
- Bani-Harouni et al., [Rewarding Doubt](https://arxiv.org/abs/2503.02623) (log-score reward; ICLR 2026)
- [DCPO](https://arxiv.org/abs/2603.09117), [CAPO](https://arxiv.org/abs/2604.12632), [confidence reward hacking](https://arxiv.org/abs/2607.04332) and the [ACL Findings 2026 study of RLVR overconfidence](https://aclanthology.org/2026.findings-acl.610.pdf) (abstracts only)
- Guo et al., [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) (calibration, ECE, temperature scaling)
- OpenAI, [GPT-4 Technical Report](https://arxiv.org/pdf/2303.08774v6) (calibration before and after post-training)
- Sharma et al., [Towards Understanding Sycophancy in Language Models](https://arxiv.org/abs/2310.13548) (ICLR 2024) and Kirk et al., [Understanding the Effects of RLHF on LLM Generalisation and Diversity](https://arxiv.org/abs/2310.06452) (ICLR 2024)
- Kalai et al., [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) (OpenAI, 2025)
- Hugging Face, [TRL GRPO trainer docs](https://huggingface.co/docs/trl/en/grpo_trainer) (default $\beta=0$)
- Turtel et al., [Outcome-based reinforcement learning to predict the future](https://arxiv.org/abs/2505.17989) (TMLR 2025; Brier reward, GRPO without std scaling, ReMax)
- [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240) (p(Yes) as a verifier score)
- Kadavath et al., [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) (P(True), P(IK))
- Müller et al., [Transformers Can Do Bayesian Inference](https://arxiv.org/abs/2112.10510) (PFNs, ICLR 2022) and Hollmann et al., [TabPFN](https://www.nature.com/articles/s41586-024-08328-6) (Nature 2025)
- Leng et al., [Taming Overconfidence in LLMs: Reward Calibration in RLHF](https://arxiv.org/abs/2410.09724) (ICLR 2025; PPO-M, PPO-C)
- Provider outputs: OpenAI, [Using logprobs](https://developers.openai.com/cookbook/examples/using_logprobs) and the [chat completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/list); Google Cloud, [Gemini GenerateContent reference](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/inference); Cleanlab, [TLM docs](https://help.cleanlab.ai/tlm/) and [FAQ](https://help.cleanlab.ai/tlm/faq/); TypeSafe, [quick start](https://docs.typesafe.ai/introduction/quickstart); on Claude, [an SDK issue](https://github.com/anthropics/anthropic-sdk-python/issues/392) and [Spice AI's Anthropic notes](https://spiceai.org/docs/components/models/anthropic)
- Tian et al., [Just Ask for Calibration](https://aclanthology.org/2023.emnlp-main.330/)
- Decision models: OpenAI, [Decisions guide](https://developers.openai.com/api/docs/guides/decisions); Cloudflare, [Introducing Clef](https://blog.cloudflare.com/clef-decision-models/) and the [clef-flash model page](https://developers.cloudflare.com/workers-ai/models/clef-flash/); Microsoft, [Microsoft-Decision-1](https://ai.azure.com/catalog/models/Microsoft-Decision-1); [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b); [Laya](https://github.com/NandhaKishorM/laya); [jevper](https://github.com/zhulinchng/jevper); LLM Gateway, [Decisions](https://docs.llmgateway.io/features/decisions) (prices and the route mapping)
- Gneiting and Raftery, [Strictly Proper Scoring Rules, Prediction, and Estimation](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) (JASA 2007)
- Independent RLCD work: Gao and Wang, [OpenJev-RLCD](https://arxiv.org/abs/2609.38850); Tegong, [JevRL](https://jevrl.github.io/output/pdf/jevrl_paper.pdf); Datawhale, [Jev Cookbook, RLCD experiment](https://datawhalechina.github.io/jev-cookbook/materials/02-nanojev/pages/docs/RLCD_EXPERIMENT/)
- Independent tests of Jev: Guo et al., [Just Ask Jev](https://arxiv.org/abs/2609.29429); Rao and Callison-Burch, [Jev vs. LLMs as Rubric Judges](https://arxiv.org/abs/2609.29769); Xu et al., [Same-Number Citation Swaps](https://arxiv.org/abs/2610.08675); Liu, [Adversarial Cues in Decision Models Used as Judges](https://arxiv.org/abs/2610.11436); Azizi et al., [The Option-Channel Attack](https://arxiv.org/abs/2610.12292)
- Qwen: QwenCloud, [Decision Model API reference](https://docs.qwencloud.com/api-reference/decision-model-api)
- Epstein, [A Scoring System for Probability Forecasts of Ranked Categories](https://journals.ametsoc.org/view/journals/apme/8/6/1520-0450_1969_008_0985_assfpf_2_0_co_2.xml) (1969)
- TypeSafe, [launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev), and docs ([AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer), [Confidence](https://docs.typesafe.ai/confidence), [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)), accessed October 2026
- Latent Space, [Jev: System One models for Prod, not God](https://www.latent.space/p/jev) (September 2026), and InfoQ, [TypeSafe AI Releases Jev](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/) (October 2026)
- [The video](https://www.youtube.com/watch?v=YGgNBcIgI4s) that started this post
