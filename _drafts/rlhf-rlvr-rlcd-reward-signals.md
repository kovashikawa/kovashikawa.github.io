---
layout: single
title: "Reinforcement learning for language models in 2026, and what System One models changed"
excerpt: "A map of how language models are trained with RL today, from RLHF and RLAIF to RLVR, process rewards and calibrated RL, with the equation each one optimizes and what System One models like Jev change."
description: "RL for language models in 2026: the objective behind RLHF, RLAIF, RLVR, GRPO, process rewards and calibrated RL, and what System One models like Jev change."
tags: [reinforcement-learning, llm, calibration, statistics, post-training, rlhf, rlaif, rlvr, grpo, system-one]
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

Most of what happens to a language model after pretraining is reinforcement learning, and what separates one recipe from another sits in two places: where the reward comes from, and how the policy is updated against it. This post goes through the main families in use today with the equation each one optimizes. They are RLHF (reinforcement learning from human feedback) and its direct-preference cousin DPO, RLAIF (from AI feedback), RLVR (with verifiable rewards) and its GRPO-style optimizers, outcome and process rewards, execution feedback, and calibrated training, meaning RLCR (calibration rewards) and RLCD (reinforcement learning for calibrated decisions).

The last family is why I wrote it. System One models such as TypeSafe's Jev do not generate text. They take a state and a list of typed questions and return a probability over the declared options. That changes what the model's action is. In the text-generating methods below, the reward scores a sampled string, or a number the model writes inside one. Here the action is the distribution itself, so a proper scoring rule can be applied to it directly, as long as it is a differentiable output of the model. What is documented is that output contract. How Jev, or any model that copies its interface, is trained is mostly unpublished, so the post shows what the contract permits in the math and then compares the training recipes other labs have disclosed.

One way to sort the families is by who consumes the output. RLHF targets a person reading a reply, so the thing to optimize is whether they approve. RLVR targets a checker, so it is whether the answer is correct. Calibrated training targets software that will act on the output, so it is whether the number attached to the answer can be trusted. The reward changes because the consumer does.

I got here from [a video on Jev](https://www.youtube.com/watch?v=YGgNBcIgI4s). TypeSafe contrasts RLCD with RLHF and RLVR in its [launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev). In the sources I read (mid-October 2026) it has not disclosed the reward or the training algorithm, and Latent Space describes it as [a novel, unpublished technique](https://www.latent.space/p/jev). So for Jev I separate three things: what is public, which is the interface; what other labs have disclosed under the same name, which is Cloudflare, the Laya project and independent authors; and what I infer. Everything else comes from papers, and where I only read an abstract I say so.

## One objective, and what separates the methods

The published methods can be read through one lens. Each maximizes expected reward while staying near a reference model:

$$
\max_{\theta}\; \mathbb{E}_{x\sim D,\; y\sim \pi_\theta(\cdot\mid x)}\big[\,R(x,y)\,\big] \;-\; \beta\,\mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\Vert\,\pi_{\text{ref}}(\cdot\mid x)\big)
$$

This is a conceptual lens, not a description of the algorithms. InstructGPT and Tülu 3 both use PPO with a learned value model. GRPO, covered below, drops the critic, clips a per-token probability ratio, and adds KL as a separate penalty. Many open GRPO recipes drop the KL term entirely. TRL's trainer now defaults to $\beta=0$ and cites Open-Reasoner-Zero, Dr. GRPO and DAPO as showing the term is not essential ([TRL docs](https://huggingface.co/docs/trl/en/grpo_trainer)). Whether RLCD has a reference policy and a KL term at all is unknown, because Jev does not generate text. What the methods do share is that everything interesting is in $R$.

The methods differ on the two axes the surveys use, where $R$ comes from and how the update is computed, and on a third that this post adds, what the model's action is.

| Method | $R$ comes from | Update | What the reward scores |
|---|---|---|---|
| RLHF | A reward model fit to human comparisons | PPO with a KL penalty | One sampled response |
| DPO | An implicit reward, the log-ratio of policy to reference | Supervised loss on preference pairs, no sampling | A fixed preference pair |
| RLAIF | A reward model fit to an LLM's soft preference labels, or the LLM's own rating | As RLHF | One sampled response |
| RLVR | A deterministic checker | PPO, or critic-free GRPO, RLOO, Dr. GRPO | One sampled response |
| Outcome and process rewards | A learned verifier of the final answer or of each step | Usually reranking; as an RL reward it is a separate choice | A sampled solution, or each step in it |
| Execution feedback (RLEF) | Running the code against tests | End-to-end RL over several turns | A multi-turn code attempt |
| RLCR | A checker plus a proper scoring rule on stated confidence | GRPO | A sampled answer and its stated confidence |
| RLCD recipes | A proper scoring rule on the returned distribution, per the recipes disclosed | Supervised loss, then sometimes REINFORCE-style RL | The whole distribution over declared options |

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

Approval has other documented costs. [Sharma et al. (2023)](https://arxiv.org/abs/2310.13548) found sycophancy across five RLHF-trained assistants. They also found that humans and preference models sometimes prefer convincing sycophantic answers to correct ones, and that on the hardest misconceptions the Claude 2 preference model chose the sycophantic response almost half the time (45%). They note that pretraining and SFT likely contribute as well. [Kirk et al. (2023)](https://arxiv.org/abs/2310.06452) found that RLHF generalises better than SFT but substantially reduces output diversity. I did not see either paper report a calibration measurement.

Approval is a fair target for an assistant, where a person reads the reply and decides what to do with it. It is a poor target when software consumes the output. Approval is not correctness, and nothing in it tells a downstream system when to doubt the answer. The model has no reason to be indifferent to how it comes across, because that is exactly what it was paid for.

The evidence on calibration is consistent with that. OpenAI reports that the pre-trained GPT-4 is highly calibrated on a subset of MMLU, and that [after post-training the calibration is reduced](https://arxiv.org/pdf/2303.08774v6). That is a statement about token probabilities and about the whole post-training process, not RLHF alone. A later study found that for RLHF-tuned models, [verbalized confidences are typically better calibrated than their token probabilities](https://aclanthology.org/2023.emnlp-main.330/), often cutting expected calibration error by a relative 50%.

## RLAIF and Constitutional AI: the labeler becomes a model

RLAIF keeps the RLHF pipeline and replaces the human comparison with a model's. [Lee et al. (2023)](https://arxiv.org/abs/2309.00267) prompt an LLM to say which of two responses is better, take the log-probabilities of the tokens "1" and "2", and apply a softmax. The result is a soft label $\hat p=P(\text{response 1 is better})$ instead of a hard choice, and the reward model is trained with cross-entropy against it. Writing the reward model's scores as $r_1,r_2$, the Bradley-Terry probability is $\sigma(r_1-r_2)$ and the loss is

$$
\mathcal{L}_r=-\,\mathbb{E}\Big[\hat p\,\log\sigma(r_1-r_2)+(1-\hat p)\,\log\sigma(r_2-r_1)\Big]
$$

which is the RLHF loss when $\hat p$ is 0 or 1. The paper states the training as cross-entropy on the softmax of the reward model's scores, and the form above is that written out. The authors also note that training on AI labels can be viewed as a form of distillation of the labeler.

They propose a second variant, direct RLAIF, which drops the reward model. The LLM rates a response from 1 to 10, the likelihood of each score token is normalized to a distribution $P(i\mid y,x)$, and the reward is the expected rating, rescaled to the range $[-1,1]$:

$$
s(y\mid x)=\sum_{i=1}^{10} i\,P(i\mid y,x)
$$

[Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) uses the same move with written principles. In the supervised stage the model critiques and revises its own outputs against the principles, and in the RL stage the preference labels come from an AI judging against them. I have read that abstract and not the full paper.

What matters for the rest of this post is the shape of the label. In both RLAIF variants the training signal starts as a probability distribution over a small closed set of tokens, "1" or "2" in one, "1" to "10" in the other, read off a model that was trained to generate text. That is the token-probability route to a typed probability. A System One model returns a distribution over declared options as its output, instead of having it recovered from token probabilities afterwards.

## RLVR: verifiable rewards replace the reward model

RLVR removes the learned proxy. Tülu 3 defines the reward as a verifier's verdict ([Lambert et al., 2024](https://arxiv.org/html/2411.15124v3)):

$$
v(x,y)=\begin{cases}\alpha & \text{if the verifier accepts } y\\ 0 & \text{otherwise}\end{cases}
$$

with $\alpha=10$ set from pilot runs and not tuned further. The training objective is the same KL-regularized form with $v$ in place of $r_\theta$. It only works where a program can decide correctness, which in Tülu 3 means math answers and checkable instruction constraints. In their 8B model, adding RLVR on top of the DPO checkpoint moved GSM8K from 84.3 to 87.6 and MATH from 42.0 to 43.7, so the reported gains are modest and targeted.

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

In a degenerate group the reward gives no gradient, and the KL term, if there is one, is the only thing left. For prompts with rare success at small $G$ that is most groups. [DAPO](https://arxiv.org/abs/2503.14476) responds by over-sampling and filtering out prompts whose accuracy is 0 or 1 ("dynamic sampling"), and reports reaching the same performance faster. Standardization has a second effect. With one success in 16, the lone success gets an advantage of about +3.9 and each failure about -0.26, so rare wins are amplified. For accuracy that concentration on the edge of the model's ability is the point.

### Critic-free variants: RLOO and Dr. GRPO

GRPO is one of several ways to get a baseline without a value network. The leave-one-out estimator in [RLOO (Ahmadian et al., 2024)](https://arxiv.org/abs/2402.14740) uses the other samples in the group as the baseline:

$$
A_i^{\text{RLOO}}=R_i-\frac{1}{G-1}\sum_{j\ne i}R_j=\frac{G}{G-1}\big(R_i-\bar R\big)
$$

The second form, a rescaled group-centered reward, is shown in the Dr. GRPO paper, and I also checked it numerically on one binary group of 8.

[Liu et al. (2025)](https://arxiv.org/abs/2503.20783) argue that GRPO's objective adds two biases relative to PPO with a Monte Carlo baseline. Dividing each completion's loss by its length $|o_i|$ gives correct short answers larger per-token updates and penalizes long wrong answers less, so wrong answers drift longer. Dividing by the group standard deviation weights questions whose rewards are nearly all 0 or all 1 more heavily, a difficulty bias. With $\ell_{i,t}$ the clipped probability-ratio term for token $t$ of completion $i$, the weight each method puts on it is

$$
\text{GRPO: }\ \frac{1}{|o_i|}\cdot\frac{R_i-\bar R}{\operatorname{std}(R)}
\qquad\qquad
\text{Dr. GRPO: }\ \frac{1}{L}\,\big(R_i-\bar R\big)
$$

where $L$ is a constant, such as the generation budget. Dr. GRPO removes both terms, and the paper reports shorter incorrect responses at similar reasoning performance. It is the same standard-deviation term that stretches the RLCR advantages below, and that Turtel et al., below, also dropped.

### What RLVR does not reward

The reward is the same whether the model was sure or guessing. In the notation of the [RLCR paper](https://arxiv.org/abs/2507.16806), $R_{\text{correctness}}=\mathbb{1}_{y\equiv y^*}$, and the authors point out that it rewards a confident correct answer and a lucky guess identically, and penalizes an abstention and a wrong answer identically. Any confidence the model states is unconstrained by the reward. They cite prior work showing models drift toward overconfidence after RL, and note that reasoning models in particular show worse calibration and more hallucination than their base models.

This is the same incentive [Kalai et al. (2025)](https://arxiv.org/abs/2509.04664) identify in evaluation. Most benchmarks grade answers as right or wrong, so guessing beats abstaining, and they argue that is one reason models hallucinate. A binary training reward has the same shape.

Other papers report the same failure. An [ACL Findings 2026 study](https://aclanthology.org/2026.findings-acl.610.pdf) finds that RLVR produces extremely overconfident models while SFT stays better calibrated, and [DCPO](https://arxiv.org/abs/2603.09117) describes calibration degeneration under RLVR. I have read those two abstracts, not the papers.

## Outcome, process and environment rewards

RLVR scores the final answer. Two other designs change where the reward attaches.

**Outcome and process reward models.** In [Lightman et al. (2023)](https://arxiv.org/abs/2305.20050), an outcome reward model (ORM) is trained to predict whether a whole solution is correct, using the final-answer check as the label, and its score is its prediction at the last token. A process reward model (PRM) predicts the correctness of each step, as one token after the step, trained by maximizing the log-likelihood of the target tokens. With $s_t$ the PRM's probability that step $t$ is correct, the loss on a solution with step labels $c_t$ is binary cross-entropy,

$$
\mathcal{L}_{\text{PRM}}=-\sum_{t}\Big[c_t\log s_t+(1-c_t)\log(1-s_t)\Big]
$$

and the solution score is the probability that every step is correct,

$$
S(\text{solution})=\prod_{t=1}^{T}s_t
\qquad\text{or}\qquad
\min_t s_t .
$$

In Lightman et al. each step is labeled positive, neutral or negative, and process supervision stops at the first incorrect step, so $s_t$ is best read as the probability that step $t$ is correct given that the earlier steps are, and the product is then the chain rule. The loss above is a simplification that counts neutral as positive. The headline result is best-of-1860 selection on a MATH subset, where the PRM solves 78.2%, the ORM 72.4% and majority voting 69.6%. The product scored 78.2% against 77.6% for the minimum, with neutral steps counted as correct. The paper calls the differences among scoring variants minor, and counting neutral as negative reverses the order (77.8% minimum against 77.4% product), so I would not read anything into product against minimum. The authors do note the product biases against solutions with more steps, which is easy to see: 50 steps that each score 0.98 give $0.98^{50}\approx0.36$. In that paper the PRM ranks samples and the policy is not trained against it, so using a PRM as an RL reward is a separate design choice and I am not describing a specific recipe.

Note what these models are. Both are probability-of-correctness classifiers trained under a log score, which is the same object as a calibrated confidence. Lightman et al. observe that their large PRM is slightly miscalibrated toward positive labels. The ORM's label is also imperfect, since the final-answer check marks a right answer reached by wrong reasoning as correct.

**Execution feedback.** [RLEF (Gehring et al., 2024)](https://arxiv.org/abs/2410.02089) trains a code model end-to-end with RL over several turns, where the model sees the results of running its code against tests and revises. The reward is grounded in execution, not in a learned scorer. The abstract reports state-of-the-art results on competitive programming with 8B and 70B models and about an order of magnitude fewer samples. I have only read the abstract, so I do not give its objective.

In RLHF, RLAIF, RLVR and these three, the policy emits text, and the reward scores a sample of it or, for DPO, a fixed pair of responses. RLCR keeps that frame and adds a probability the model writes inside its text. The System One section changes it.

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

Reported results, from the paper's own abstract and introduction. On in-domain tasks RLCR matches RLVR accuracy while cutting ECE from 0.37 to 0.03 on HotpotQA and from 0.26 to 0.10 on a collection of math datasets. Out of domain, RLVR made calibration worse and RLCR improved it. The authors are explicit that out-of-domain calibration error is still high in absolute terms, and that models can still assign high confidence to contradictory answers. Besides RLVR the paper compares against classifiers trained to output a confidence for the RLVR model's answers, a linear probe, and the answer-token probability. In-domain ECE on HotpotQA is 0.07 for the BCE classifier, 0.09 for the Brier classifier and 0.10 for the probe, against 0.03 for RLCR, and the out-of-domain averages are 0.24, 0.33 and 0.38 against 0.21. So the margin over a trained post-hoc classifier is much smaller than the margin over plain RLVR (0.37), especially out of domain. I did not see a comparison with temperature scaling of the verbalized confidence.

Group-relative training also gets more signal. In an all-correct group of 8 where the model states different confidences, binary advantages are all zero. Under RLCR the rewards differ, so the advantages are nonzero and the model is pushed toward higher confidence when every sample is right. These advantages use the population standard deviation. The sample standard deviation shrinks every one of them by about 6.5%, a factor of $\sqrt{7/8}$.

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

- ECE at $k=1$ is zero at the population level. The 0.002 is finite-sample noise from 15-bin ECE, which is biased upward in small samples, and it ranged from about 0.002 to 0.004 across 30 seeds.
- $k>1$ is overconfident only on average. For $p<0.5$ it pushes $q$ below $p$. Most of the Beta mass sits above 0.5, so the mean $q$ at $k=5$ is 0.69 against 0.62 accuracy.
- The log loss of 1.067 at $k=5$ is inflated partly by clipping $q$ at $10^{-6}$. That is a separate issue from the incentive problem above.

The baseline this table ignores is post-hoc recalibration. Because $k$ is a temperature ($q=\sigma(\operatorname{logit}(p)/T)$ with $T=1/k$), a single parameter fit on held-out data, which is the temperature scaling Guo et al. describe, would remove every row of miscalibration here without touching the model's weights. Any monotone remap also leaves ranking, and so AUROC, unchanged. Training-time calibration earns its cost when the model's confidence is poorly ordered, which a remap cannot fix, or when confidence has to be produced alongside the reasoning. I would treat the head-to-head as an open question.

## System One and RLCD: what changed, what is public, what I infer

TypeSafe's [primer](https://docs.typesafe.ai/introduction/machine-learning-primer) describes RLCD in three points. The model does not generate text. It returns decisions and probabilities. Higher probability should correspond to a greater chance the answer is correct. Their docs define calibration as outcomes assigned 0.2 happening about 20% of the time, and say it describes groups of predictions, not any single answer. The docs also credit cofounder Diogo Almeida with co-inventing RLHF. He is an author of InstructGPT (Ouyang et al., 2022). He is not an author of Christiano et al. (2017), the paper that introduced deep RL from human preferences.

In my searches for a TypeSafe technical report, paper, repository or model card I found no publication of the reward itself. [OpenJev-RLCD](https://arxiv.org/abs/2609.38850) says TypeSafe discloses no training algorithm, and the [Datawhale cookbook](https://datawhalechina.github.io/jev-cookbook/materials/02-nanojev/pages/docs/RLCD_EXPERIMENT/) calls its own formula \"an independent research candidate, not a recovered official recipe\". My reading, and it is only that, is that the output contract makes the problem simpler than RLCR's. RLCR has the model choose an answer and attach a confidence, so the reward must couple them, and a bounded rule is needed to stop the model trading accuracy for score. A model that emits the full distribution $q(\cdot\mid x)$ over a fixed answer set can be scored directly with a proper rule against the observed label, with no answer-versus-confidence trade, which is why a log score would not hit the problem above. For the log score,

$$
\mathbb{E}_{y\sim p}\big[\log q(y)\big]=-H(p)-\mathrm{KL}(p\,\Vert\,q)
$$

is maximized at $q=p$. If that is roughly the shape of RLCD, it sits close to classical probabilistic classification, and the new part is the interface. Natural-language state in, typed questions (Choice, Score, Noul) out, in a single query. TypeSafe's launch post describes parallel sampling, and I do not know whether the probabilities come from a softmax head or from sample frequencies, which changes what a scoring rule would be applied to. I do not know the reward, and I would not cite any of this as TypeSafe's method. Published work on adjacent systems narrows the options, and I collect it in \"Clues from adjacent work\" below.

What is public is the interface, and it makes calibration usable. A Choice answer returns a probability per option, and TypeSafe reports a `confidence` derived from that distribution. For $n$ options with top probability $p_{\max}$ the [documented formula](https://docs.typesafe.ai/confidence) is

$$
\text{confidence}=\frac{p_{\max}-1/n}{1-1/n}
$$

and for a Noul with probability $p$ it is $\lvert 2p-1\rvert$. Only the top probability counts, so $(0.6,0.3,0.1)$ and $(0.6,0.2,0.2)$ both come out at 0.4. This is a rescaled margin over chance, not the probability of being right. A confidence of 0.9 means $p_{\max}=0.95$ for two options and 0.933 for three, through $p_{\max}=\text{confidence}\cdot(1-1/n)+1/n$. Decision thresholds belong on the probabilities.

A typed output cannot be a malformed type, but it can still be a wrong valid value. [InfoQ](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/) reports that objection from Hacker News against TypeSafe's "can't hallucinate" line, and quotes Armin Ronacher saying the design "delegates the hallucination problem a little bit to the user".

### What changes in the math

For the text-generating methods the action is a token sequence $y\sim\pi_\theta(\cdot\mid x)$, and the reward is evaluated on a sample. The gradient is the score-function estimator with a baseline $b$,

$$
\nabla_\theta J=\mathbb{E}_{y\sim\pi_\theta}\Big[\big(R(x,y)-b\big)\,\nabla_\theta\log\pi_\theta(y\mid x)\Big]
$$

and its variance comes from sampling $y$. GRPO and RLOO use different baselines $b$, and GRPO also rescales by the group standard deviation.

A System One model maps a state $x$ and $K$ declared options to a distribution $q_\theta(x)\in\Delta^{K-1}$. The action is the distribution. With $\ell$ a loss and $y$ the observed outcome, the objective is

$$
J(\theta)=-\,\mathbb{E}_{(x,y)}\big[\ell\big(q_\theta(x),\,y\big)\big]
$$

Two conditions make this simple. First, $q_\theta$ has to be a differentiable output of the model, a head, and not a frequency over sampled answers or a function of a sampled rationale. Second, the label $y$ has to be exogenous, the way a forecast target is, and not something the model's own output changes. When both hold there is nothing to sample on the model side, and the gradient is ordinary backpropagation through $q_\theta$. The data $(x,y)$ are still a sample, which is why this looks like supervised learning with a proper loss. Whether any RL is involved is the open question.

The first condition fails in two places that matter below. If $q$ is a frequency over $n$ sampled answers, then for a binary question the expected Brier loss is inflated by the variance of the estimate, $p(1-p)/n$, and JevRL's duplicate penalty exists to remove that bias. And [OpenJev-RLCD](https://arxiv.org/abs/2609.38850) scores an answer distribution that comes after a sampled rationale, so its second stage does need a sampling-based estimator for the rationale.

If $\ell$ is strictly proper, the expected loss is minimized only at the true conditional distribution $p$,

$$
\arg\min_{q}\ \mathbb{E}_{y\sim p}\big[\ell(q,y)\big]=p
$$

which is the property the logarithmic loss $-\log q_y$, the Brier loss $\lVert q-e_y\rVert^2$ and the spherical score share ([Gneiting and Raftery, 2007](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437)). This is a statement about the population optimum. A finite model trained on finite data does not land there. [Guo et al. (2017)](https://arxiv.org/abs/1706.04599) show that networks trained on cross-entropy come out overconfident, and the fitted temperature of 2.41 for Kev-4B below says the same about one disclosed head. So the objective targets calibration. It does not deliver it, and held-out recalibration stays part of any recipe.

| | Free-text RL (RLHF, RLVR, RLCR) | System One, what the output type permits |
|---|---|---|
| Action | A token sequence $y\in V^*$ | A distribution $q$ over $K$ declared options |
| What is scored | One sampled $y$ | The whole $q$ |
| Reward needs | A verdict or score on the sample | An observed label $y$ for each training state |
| Gradient | Score-function estimator with a baseline | Backpropagation through $q$ if it is a differentiable head and $\ell$ is differentiable. Otherwise sampling estimators come back |
| Where calibration comes from | Stated text, shaped by the reward | The objective's target at the population optimum, then finite-sample fit and recalibration |

The cost of the second column is the label. RLVR on math also needs a reference answer for every prompt, but code and constraint checkers need no label at all, and a free-text answer has many valid strings, so a checker has to judge equivalence. A typed option has one label per state. The price is that the label must be the true outcome for every training state, and for real decisions that outcome is often delayed, noisy or counterfactual, since a blocked action never shows what would have happened. The recipes below differ mostly in how they respond to that.

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

### Recipes other labs have disclosed

Jev is no longer alone. Since its launch, other groups have shipped models with the same interface, a state plus typed questions in and probabilities out. They differ in how much they say about training, and that is what is useful here, because it shows what the recipes behind the name contain. The table lists only what each project says about training. I read each project's own pages.

| Recipe | Backbone | Supervised stage | RL stage |
|---|---|---|---|
| [Cloudflare Clef and Clef-flash](https://blog.cloudflare.com/clef-decision-models/) | Frozen Qwen3.8-27B and Qwen3.5-9B, routing head, rank-256 LoRA | Label-smoothed cross-entropy plus a Brier loss | "RLCD" as a secondary stage: partial credit for adjacent ordinal choices, a reward for exactly right records, a reference penalty |
| [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b) | Frozen Qwen3.5-4B-Base, LoRA rank 16, pointer head | Cross-entropy over the options in four stages, then one temperature $T=2.41$ fitted by negative log-likelihood | None |
| [Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large encoder, 421M, no text generation | Soft cross-entropy against a teacher | Group-mean baseline, noise on the logits, reward from strictly proper scores (log, spherical, ranked score for ordinal questions) |
| [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions), [Microsoft-Decision-1](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/), [QwenCloud](https://docs.qwencloud.com/api-reference/decision-model-api) | Microsoft: Qwen3.5-9B. Others not stated | Microsoft: public datasets plus synthetic data. Others not stated | None stated in the pages I read |

The recipes use three losses between them, with $K$ options, a reported distribution $q$ and an observed label $y$.

$$
\mathcal{L}_{\text{CE}}=-\sum_{k=1}^{K}\Big[(1-\varepsilon)\,\mathbb{1}[k=y]+\tfrac{\varepsilon}{K}\Big]\log q_k
\qquad
\mathcal{L}_{\text{Brier}}=\sum_{k=1}^{K}\big(q_k-\mathbb{1}[k=y]\big)^2
$$

Cross-entropy with label smoothing $\varepsilon$ is the log loss against a softened target. It is not proper. Its population minimizer is $(1-\varepsilon)p+\varepsilon/K$, not $p$, so it shrinks probabilities toward uniform on purpose, and a weighted sum of it with the Brier loss is not proper either. The Brier loss alone is the quadratic score and is proper. Laya adds the spherical score, $S_{\text{sph}}(q,y)=q_y/\lVert q\rVert_2$, and for ordered levels the ranked probability score,

$$
\mathrm{RPS}(q,y)=\sum_{k=1}^{K-1}\Big(\sum_{j\le k}q_j-\mathbb{1}[y\le k]\Big)^2 .
$$

The RPS charges more for probability placed far from the true level than for probability placed next to it, which is what "partial credit to adjacent ordinal choices" would amount to in a reward. Gneiting and Raftery's [review of scoring rules](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) covers the propriety of the logarithmic, spherical and quadratic scores. The RPS is [Epstein's (1969)](https://journals.ametsoc.org/view/journals/apme/8/6/1520-0450_1969_008_0985_assfpf_2_0_co_2.xml) score for ranked categories, the sum of squared differences between cumulative forecast and cumulative observed probabilities. The last cumulative term is always zero, which is why the sum stops at $K-1$, and some authors divide by $K-1$.

The RL stage in these recipes deserves a closer look. Laya's card describes REINFORCE with a group-mean baseline,

$$
\nabla_\theta J \approx \mathbb{E}\Big[\big(R-\bar R\big)\,\nabla_\theta \log \pi_\theta\Big],
$$

with $R$ a scoring rule applied to the reported distribution. Here is my own inference. A proper score is a differentiable function of $q$, so when $q$ is a deterministic head the exact gradient is available without sampling. For the log score and softmax logits $z$ it is $\nabla_z \log q_y = e_y - q$, the same gradient as cross-entropy. REINFORCE then estimates something supervised training already computes, with more variance. If exploration noise is injected on the logits, as Laya's card describes, REINFORCE is unbiased for a different objective, the score averaged over that noise, which acts like a smoothing regularizer. I have not read Laya's training code, so I do not know which of the two it is. A reason to pay for it is a reward that is not differentiable, and "fully precise record outputs" in Cloudflare's description is that kind of reward, an exact match across several fields. This is why I read the supervised stage as the load-bearing one and RL as the part that handles what a loss cannot express. An independent control experiment, below, finds a sampled proper reward no clearly better than the direct losses. It is a reading of other people's recipes, not a statement about TypeSafe's.

Two things follow.

1. **Supervised loss first.** Cloudflare uses label-smoothed cross-entropy and Brier before RL. Kev uses no RL at all and reaches usable probabilities with cross-entropy plus one temperature, the post-hoc fix from [Guo et al.](https://arxiv.org/abs/1706.04599). A fitted $T$ above 1 flattens a softmax, so I read that as the raw head having been overconfident.
2. **"RLCD" is now several recipes.** Cloudflare's rewards ordinal closeness, exact records and staying near a reference model. Laya's rewards a proper score on the whole distribution. The next subsection adds independent ones. None says it is TypeSafe's, and I found no public TypeSafe recipe.

### Independent RLCD implementations

Since the launch, authors with no tie to TypeSafe have published their own implementations of RLCD. I read the abstracts and, for OpenJev-RLCD, the algorithm section. I did not reproduce any result.

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

One caveat on the one-hot form. With $M$ one-hot samples drawn from a distribution $q$ and observed label $y$, the expected reward works out to $2q_y-1-(1-\tfrac1M)\lVert q\rVert^2-\tfrac1M$. Against a true distribution $p$ it is maximized not at $p$ but at the projection of $p/(1-1/M)$ onto the simplex, so it sharpens at small $M$ and is proper only as $M\to\infty$. For $p=(0.5,0.3,0.2)$ and $M=2$ the maximizer is $(0.667,0.267,0.067)$. I checked this numerically. JevRL's $\mathbb{1}[A=B]$ term is the unbiased estimate of $\lVert q\rVert^2$, which is why its expectation is exactly one minus the Brier score.

### Jev as a component inside RL

Everything above puts a calibrated distribution at the end of training. One paper puts it inside an RL loop. [Ma et al. (2026)](https://arxiv.org/abs/2610.11692) use a frozen, untrained Jev at three positions in a standard agent's training, as a reference policy, an exploration judge and a replay rater, on nine MiniGrid tasks and three Atari games. They report that training with Jev outperforms a standard learner, including where the learner makes no progress alone. Their analysis of what each RL object demands of the answers it consumes concludes that Jev can fill all of them except the cardinal use of a value function. My reading of that exception is that a probability over options can rank actions but does not give a return in the units a value function needs. I read the abstract only.

### Where Jev sits on this map

What is public about Jev is the interface: typed questions in, a distribution over declared options out, no generated text. That puts its action in the second column of the comparison table in \"What changes in the math\". What I cannot place is the training. The disclosed recipes run from a supervised proper loss only (Kev), through supervised then RL (Cloudflare), to RL with a proper-score reward (Laya, OpenJev-RLCD). My read, as inference, is that Jev is closest to a typed-head model trained against a proper score, possibly with an RL stage for rewards a loss cannot express. Cloudflare, Laya and independent authors each use the name RLCD for their own recipes, and nothing I read says any of them matches TypeSafe's.

Two public facts would change my read. A technical report with the loss would settle it directly. A comparison of calibration before and after the RL stage would show whether that stage adds anything over supervised training on the same labels.

## How each one gets gamed

**RLHF.** The reward model is a proxy, and pushing the policy hard against it eventually costs real quality (Gao et al., above). A rater who cannot check a claim can still prefer the confident phrasing, so a proxy trained on those choices can reward sounding right.

**RLAIF.** The reward inherits whatever the labeler model gets wrong. Lee et al. describe training on AI labels as a form of distillation, so a systematic bias in the labeler becomes a systematic bias in the reward model. When an LLM judge is the live reward, as in direct RLAIF, the policy can also learn the judge's blind spots, and [Liu (2026)](https://arxiv.org/abs/2610.11436) shows one such sensitivity in a decision model used as a judge, described below.

**RLVR.** A verifier is only as good as its coverage. Typical failures, from general experience and not from a source in this post, are answers that match the checker's format without being right, special-casing the test cases, and false negatives that punish correct answers. Execution rewards also run model-written code, so the reward step needs a sandbox. It also says nothing about confidence. The optimizer adds its own bias: GRPO's length and standard-deviation normalization, which Dr. GRPO removes.

**Process rewards.** The product score penalizes long solutions even when every step is individually likely, as in the 50-step example above, and the large PRM in Lightman et al. leans toward calling steps correct. An outcome label can mark a right answer reached by wrong reasoning as correct.

**RLCR.** Properness is a statement about the optimum, not about optimization. A [paper on confidence rewards](https://arxiv.org/abs/2607.04332) describes confidence reward hacking, where a poorly designed scheme lets the model gain by answering wrongly so that it can be confident the answer is wrong, and reports that it occurs on practical datasets. Two more papers put the problem in the optimizer. DCPO reports a gradient conflict between maximizing accuracy and minimizing calibration error, and [CAPO](https://arxiv.org/abs/2604.12632) argues that GRPO's uncertainty-agnostic advantage degrades relative calibration. Again, I have read abstracts.

**RLCD.** Unknown for TypeSafe, because I found no published reward. Partly known for the others, since Cloudflare, Laya and independent authors describe recipes. Two caveats come from outside the reward. Calibration is an average-case property, not a guarantee against crafted input. Two independent studies, which I read as abstracts only, show how that fails. [Liu (2026)](https://arxiv.org/abs/2610.11436) found that adding one colon to a candidate raised Jev's false acceptance as a judge from 1.0% to 26.0% on 200 clusters, but only when the request's JSON keys were sorted, and both variants were rejected under insertion order. The cause is unmeasured. [Azizi et al. (2026)](https://arxiv.org/abs/2610.12292) did not test Jev. On seven open-weight typed models used as agent guardrails, they report allow-or-block accuracy of 36% to 72% against a 50% chance level, six lines of irrelevant log text raising fail-open from 0% to 63% on one synthetic suite, and a misleading option name raising it to 93% to 100% on the four models that put the label in their input. They also report that escalating the least confident decisions does not help, because a decision an attack has reversed is no less confident than the one it replaced. For guardrails and routing the state often contains untrusted text, and so do the option names and definitions, which are an input channel too. Treat all of it as hostile. Azizi et al. report that parsing a policy field into a typed value and applying a deterministic rule reached 100% accuracy on their six policies, which also makes the model unnecessary, and they conclude these models should not be the component that decides. Typed outputs also do not prevent wrong valid values, as above.

## Where each one belongs

Calibration is what lets software choose a threshold with arithmetic. Say an automated action costs $c_{FP}$ when it fires wrongly and skipping a true positive costs $c_{FN}$. If $p$ is a calibrated probability, the expected cost of acting is $(1-p)\,c_{FP}$ and of not acting is $p\,c_{FN}$, so act when

$$
p>\frac{c_{FP}}{c_{FP}+c_{FN}}
$$

With illustrative numbers, a wrongly issued refund costing 50 and a missed one costing 5, the threshold is $50/55\approx0.91$. In the video that prompted this post, the example cutoff on the returned probability was 0.9, close to this.

Two assumptions sit under the rule. Correct actions and correct skips cost nothing, and the probability is calibrated on your own traffic. Add a human-review cost $c_h$ and a review band exists only when $c_h < c_{FP}c_{FN}/(c_{FP}+c_{FN})$, which is 4.55 here. At $c_h=2$ you review between $p=0.40$ and $p=0.96$ and act automatically only above $1-c_h/c_{FP}=0.96$. Thresholded decisions also depend on how well the score ranks right answers above wrong ones (AUROC, risk-coverage curves), not only on calibration.

Both assumptions also require that the input is not adversarial. Calibration is an average-case property over the data the model was fit on. If part of the state is written by someone the decision is meant to guard against, that person influences $p$ too. In the guardrail study above, sending low-confidence cases to review does not catch a reversed decision, so a review band like this one helps with honest uncertainty and not with manipulation.

Calibration is also conditional on the task and the data distribution. An independent study of Jev across 44 benchmarks has the same shape, with a pooled ECE of 0.047 but a median per-benchmark ECE of 0.168 ([Guo et al., 2026](https://arxiv.org/abs/2609.29429), abstract only). A model calibrated on a set that is half positives can be miscalibrated when prevalence falls to 5%, and RLCR's own out-of-domain error stays high. Check reliability on a labeled held-out sample from your own traffic, and again after any model version change. InfoQ's summary of TypeSafe's docs advises pinning a version rather than using the moving alias. A threshold chosen on a miscalibrated score is a guess with extra steps.

| Method | Use it for | Evidence in this post | Watch for |
|---|---|---|---|
| RLHF | Open-ended assistants, tone, refusals, anything with no ground truth | InstructGPT's 1.3B model preferred over 175B GPT-3 by labelers | Reward model over-optimization, and calibration loss after post-training |
| RLAIF | Preference data at scale when a strong labeler model exists | Soft labels from the logprobs of "1" and "2"; direct RLAIF needs no reward model | Labeler bias passes into the reward |
| RLVR | Math, code, formats, constraints a program can check | Tülu 3 gains on GSM8K, MATH, IFEval | Overconfidence, zero-gradient prompts, only works with a verifier |
| Process reward models | Step-structured reasoning, mainly to rank or filter samples | Product of step scores, 78.2% in best-of-1860 on a MATH subset | Penalizes long solutions; step labels are costly |
| RLEF | Code and tool tasks where running the output gives feedback | State-of-the-art on competitive programming with 8B and 70B models (abstract) | Needs executable tests |
| RLCR | Reasoning QA where the stated confidence drives a decision, plus test-time confidence-weighted voting | ECE 0.37 to 0.03 on HotpotQA | Out-of-domain calibration still imperfect, confidence reward hacking |
| RLCD | Routing, guardrails, classification, reranking, verification steps in a pipeline | TypeSafe's cookbooks, which are vendor demos | Not generative, average-case calibration, adversarial or untrusted input, wrong valid values, jagged on counting, dates and large noisy state |

Two of those rows deserve a sentence each. RLCR's confidence is useful at inference time. The paper's confidence-weighted majority voting and max-confidence selection both improve accuracy and calibration as compute scales. For RLCD, TypeSafe's own [jaggedness page for jev-1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13) says it is not a calculator, reads literally, and degrades when the state is full of irrelevant detail. I have not tested Jev myself, and its cookbook numbers are the vendor's.

Calibrated confidence has uses beyond the table. These apply to calibrated confidence in general, not to any one method:

- **LLM-as-judge.** A calibrated judge score can be thresholded or averaged sensibly. An uncalibrated one cannot.
- **Selective prediction.** Answer or abstain, evaluated with risk-coverage curves and not ECE alone.
- **Test-time compute.** Spend more samples where confidence is low.
- **Tool-call gating in agents.** Allow, ask, or block based on a probability, only where the state is trusted. See the guardrail study above.
- **Retrieval-augmented answers.** Answer, or say the sources do not contain it.

## The methods in short

Each family answers the question of who the output is for. RLHF optimizes a model of what people prefer. RLAIF distills a model's preferences in their place. RLVR and RLEF optimize what a program accepts. A process reward model optimizes a learned verifier of each step. RLCR adds a proper scoring rule to a verifiable reward so that stated confidence means something. System One models change the action, from a sampled string to a distribution over declared options, so a proper score can be the objective directly and the label replaces the verdict. How much of what TypeSafe calls RLCD is reinforcement learning, and how much is supervised training on a proper score, is the part I found no outside check of.

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
| $\hat p$ | A soft preference label from an AI labeler (RLAIF) |
| $s_t$ | A process reward model's probability that step $t$ is correct |
| $o_i$, $\bar R$, $A_i$ | Completion $i$ in a group, the group mean reward, its advantage |
| $\ell$, $S$, $\Delta^{K-1}$ | A proper loss to minimize, written $S$ in the RLCR section and $\ell$ in the System One section, and the simplex of distributions over $K$ options. The spherical score $S_{\text{sph}}$ is the one score to maximize |
| ECE, Brier, RPS | Expected calibration error, the quadratic score, the ranked probability score |

$K$ means two things. In the RLHF section it is the number of responses a labeler ranks, as in InstructGPT. From the calibration sections on it is the number of answer options. In the same way, $\theta$ parameterizes the reward model in the RLHF section, following InstructGPT, which uses $\phi$ for the policy, and it parameterizes the policy everywhere after that.

## Reproduce the numbers

The snippet reproduces the computed tables in this post: the Bradley-Terry values, the degenerate-group table, the RLCR expected reward, the all-correct-group advantages and the toy simulation. Figures quoted from papers come from those papers. The JevRL and OpenJev identities, the RLOO rescaling and the threshold arithmetic are not in it, and I checked them separately. It is about 40 lines of NumPy.

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
- Tian et al., [Just Ask for Calibration](https://aclanthology.org/2023.emnlp-main.330/)
- Decision models: OpenAI, [Decisions guide](https://developers.openai.com/api/docs/guides/decisions); Cloudflare, [Introducing Clef](https://blog.cloudflare.com/clef-decision-models/) and the [clef-flash model page](https://developers.cloudflare.com/workers-ai/models/clef-flash/); Microsoft, [Microsoft-Decision-1](https://ai.azure.com/catalog/models/Microsoft-Decision-1); [Kev-4B](https://huggingface.co/jaredpalmer/kev-4b); [Laya](https://github.com/NandhaKishorM/laya); [jevper](https://github.com/zhulinchng/jevper); LLM Gateway, [Decisions](https://docs.llmgateway.io/features/decisions) (prices and the route mapping)
- Gneiting and Raftery, [Strictly Proper Scoring Rules, Prediction, and Estimation](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) (JASA 2007)
- Independent RLCD work: Gao and Wang, [OpenJev-RLCD](https://arxiv.org/abs/2609.38850); Tegong, [JevRL](https://jevrl.github.io/output/pdf/jevrl_paper.pdf); Datawhale, [Jev Cookbook, RLCD experiment](https://datawhalechina.github.io/jev-cookbook/materials/02-nanojev/pages/docs/RLCD_EXPERIMENT/)
- Independent tests of Jev, cited for one result each, abstracts only: Guo et al. (2026), [Just Ask Jev](https://arxiv.org/abs/2609.29429); Liu (2026), [Adversarial Cues in Decision Models Used as Judges: The Role of Request Presentation](https://arxiv.org/abs/2610.11436); Azizi et al. (2026), [One Word Opens the Gate: The Option-Channel Attack on Typed Decision Models as Agent Guardrails](https://arxiv.org/abs/2610.12292)
- Qwen: QwenCloud, [Decision Model API reference](https://docs.qwencloud.com/api-reference/decision-model-api)
- Lee et al., [RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267) (soft labels from "1"/"2" logprobs, direct RLAIF) and Bai et al., [Constitutional AI](https://arxiv.org/abs/2212.08073) (abstract only)
- Ahmadian et al., [Back to Basics: Revisiting REINFORCE Style Optimization](https://arxiv.org/abs/2402.14740) (RLOO) and Liu et al., [Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783) (Dr. GRPO)
- Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (ORM, PRM, product score) and Gehring et al., [RLEF](https://arxiv.org/abs/2410.02089) (abstract only)
- Ma et al., [Can Jev be Your Q or Policy in Reinforcement Learning?](https://arxiv.org/abs/2610.11692) (abstract only)
- Epstein, [A Scoring System for Probability Forecasts of Ranked Categories](https://journals.ametsoc.org/view/journals/apme/8/6/1520-0450_1969_008_0985_assfpf_2_0_co_2.xml) (1969)
- TypeSafe, [launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev), and docs ([AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer), [Confidence](https://docs.typesafe.ai/confidence), [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)), accessed October 2026
- Latent Space, [Jev: System One models for Prod, not God](https://www.latent.space/p/jev) (September 2026), and InfoQ, [TypeSafe AI Releases Jev](https://www.infoq.com/news/2026/10/typesafe-ai-jev-released/) (October 2026)
- [The video](https://www.youtube.com/watch?v=YGgNBcIgI4s) that started this post
