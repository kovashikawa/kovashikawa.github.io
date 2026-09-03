---
layout: single
title: "Beyond Pearson's rho: measuring dependence the modern way"
date: 2026-09-02 12:00:00 -0400
excerpt: "Pearson says 0, but X and Y are dependent. A guide to distance correlation, Chatterjee's xi, HSIC, KSG MI, and tail dependence, with rerunnable benchmark."
description: "Pearson says 0, but X and Y are dependent. A guide to distance correlation, Chatterjee's xi, HSIC, KSG MI, and tail dependence, with rerunnable benchmark."
tags: [statistics, correlation, dependence, python, benchmark]
categories: [statistics]
author_profile: true
toc: true
toc_sticky: true
read_time: true
mathjax: true
---

[Last time]({% post_url 2025-07-22-lack-of-correlation-not-independence %}) I showed the canonical trap: $X \sim \mathcal{N}(0,1)$, $Y = |X|$, and Pearson's $\rho$ reports about 0 while $X$ fully determines $Y$. The one-line takeaway was "use a nonlinear measure." This post is the follow-through: which measures, what they actually guarantee, and what they miss.

Every number below comes from a runnable benchmark against eight synthetic datasets, seeded and reproducible. The companion repo is [kovashikawa/correlation-models](https://github.com/kovashikawa/correlation-models).

## The benchmark table

| Measure | linear | quadratic | abs | sine | circle | cross | independent | heavy_tail |
|---|---|---|---|---|---|---|---|---|
| Pearson | 0.981 | -0.017 | -0.015 | 0.004 | 0.004 | -0.020 | -0.008 | 0.758 |
| Spearman | 0.978 | -0.014 | -0.014 | 0.026 | 0.003 | -0.014 | -0.005 | 0.803 |
| Kendall | 0.874 | -0.011 | -0.011 | 0.023 | 0.000 | -0.008 | -0.003 | 0.620 |
| Chatterjee xi | 0.812 | 0.999 | 0.999 | 0.997 | 0.254 | 0.229 | 0.012 | 0.463 |
| Distance corr | 0.971 | 0.543 | 0.559 | 0.253 | 0.197 | 0.313 | 0.019 | 0.775 |
| HSIC | 0.088 | 0.045 | 0.051 | 0.008 | 0.019 | 0.033 | 0.000 | 0.044 |
| KSG MI | 1.651 | 6.102 | 6.378 | 4.716 | 5.167 | 6.197 | 0.012 | 0.579 |
| Tail dep (q=0.95) | 0.838 | 0.498 | 0.498 | 0.090 | 0.000 | 0.480 | 0.046 | 0.474 |

The first three rows are the classical toolkit, and they all say "no dependence" on quadratic, |X|, sine, circle, and cross. The modern measures say "obviously dependent." That is the whole problem, quantified.

Two things to notice before the details:

1. **The cross column is the honest one.** Cross is $Y = X \cdot W$ with $W$ a Rademacher sign: $Y$ is literally $\pm X$, completely determined by $X$ up to sign. Pearson, Spearman, and Kendall all report near zero because the sign symmetry cancels, while Chatterjee xi (0.229) and distance correlation (0.313) flag the dependence. It is the cleanest possible counterexample to "zero correlation means nothing is going on."
2. **HSIC and KSG MI are on different scales.** HSIC is positive but unbounded; MI is in nats and depends on marginal entropy. Use them as detectors and for ranking, not as comparable strengths.

## Distance correlation

Szekely, Rizzo and Bakirov (2007) introduced distance correlation to fix exactly this blind spot. The idea: independence is equivalent to the joint characteristic function factoring into the product of marginals. Distance covariance is a weighted norm on exactly that difference, and the estimator falls out as a double-centering of pairwise distance matrices:

$$\operatorname{dCor}(X,Y) = \frac{\operatorname{dCov}(X,Y)}{\sqrt{\operatorname{dCov}(X,X)\,\operatorname{dCov}(Y,Y)}}.$$

The property that matters: $\operatorname{dCor} = 0$ if and only if $X$ and $Y$ are independent, for distributions with finite first moments, in any dimension. Pearson cannot make that claim. In the bivariate normal case dCor is a deterministic function of $|\rho|$ and never exceeds it.

Cost: $O(n^2)$ memory and time, because of the distance matrices. Fine at 10k rows, painful at 10M.

## Chatterjee's xi

Chatterjee (2021) answered with a coefficient that is almost absurdly simple. Rank X, reorder Y's max-ranks by X, and measure how much adjacent ranks jump:

$$\xi_n(X,Y) = 1 - \frac{A_1}{C_U}, \qquad A_1 = \frac{1}{2n}\sum_{i=1}^{n-1}\left|\frac{r_{i+1}}{n} - \frac{r_i}{n}\right|, \qquad C_U = \frac{1}{n}\sum_{i=1}^{n} g_i(1-g_i),$$

where $r_i$ are max-ranks of $Y$ reordered by $X$ and $g_i$ the max-ranks of $-Y$. With no ties this collapses to $\xi = 1 - \frac{3}{n^2-1}\sum_{i=1}^{n-1}|r_{i+1} - r_i|$. $\xi = 0$ iff independence, $\xi = 1$ iff $Y$ is a measurable function of $X$. Computes in $O(n \log n)$ and is completely nonparametric.

Two honest footnotes. Finite-sample $\xi$ under independence is slightly negative on average, so small negative values are expected, not bugs. And $\xi(X, Y)$ is asymmetric by construction: it measures "how well Y behaves as a function of X," which the paper argues for deliberately.

## HSIC

HSIC comes from the kernel methods literature (Gretton et al. 2005) and became the workhorse of kernel feature selection (Song et al. 2012). Map each variable into a reproducing kernel Hilbert space with a characteristic kernel (RBF here), and take the Hilbert-Schmidt norm of the cross-covariance operator between the two embeddings. HSIC = 0 iff independence, no density estimation anywhere in the pipeline. That is why it is the workhorse of kernel feature selection, where estimating densities in high dimensions is a nonstarter.

Bandwidth choice matters. The implementation in the companion repo uses the median-distance heuristic, which is the standard default.

## KSG mutual information

Mutual information $I(X; Y) = 0$ iff independence, full stop. The KSG estimator (Kraskov, Stogbauer and Grassberger 2004) is a k-nearest-neighbor scheme that adapts its resolution in both margins, which fixes the classic histogram-bin problems. It is what scikit-learn's `mutual_info_regression` uses under the hood.

The caveat: MI values depend on marginal entropies, so "1.6 nats" on linear data and "6.1 nats" on |X| are not comparable strengths. It answers "is there dependence" decisively, and "how strong" only loosely.

## MIC

The maximal information coefficient (Reshef et al. 2011) maximizes normalized mutual information over all grid binning schemes, capped by sample size. It made a splash in Science for "detecting novel associations in large data sets" and is the measure most people name when they want "the nonlinear correlation."

The honest footnote: the equitability claims were contested, with mathematical arguments showing the proposed definition of equitability is impossible for any nontrivial measure (Kinney and Atwal 2014), and follow-up work found the simulation evidence artifactual. Treat MIC as one more detector, not a calibrated strength scale. `minepy` is the reference implementation; it is not in this repo's benchmark because its API is a separate wheel, but the claim in the table is covered by distance correlation and KSG MI, which dominate it on power anyway.

## Tail dependence

All of the above measure dependence across the whole distribution. Risk work cares about the tails specifically: given that one asset is in its 95th percentile, how likely is the other to be too? Tail dependence coefficients go back to Sibuya (1959) and Joe (1993).

$$\lambda(q) = P(F_Y(Y) > q \mid F_X(X) > q).$$

This is where the Gaussian copula earns its infamy. For any correlation $\rho < 1$, the Gaussian copula has exactly zero tail dependence: extremes never cluster, no matter how correlated the middles are. If your risk model is Gaussian-copula shaped and you feed it Pearson correlations, you are asserting away joint tail risk by construction. The empirical estimator in the benchmark shows the difference plainly: independent data sits at 0.046, the cross case at 0.480, and the linear (Gaussian-ish) case at 0.838, which converges to zero only very slowly as q approaches 1. The heavy-tail case sits at 0.474.

## When to use what

- Bivariate, monotone, want a sign: Spearman or Kendall.
- Bivariate, any shape, want a [0, 1] strength: distance correlation or Chatterjee xi.
- High-dimensional or vector-valued: distance correlation, HSIC.
- Nonlinear feature screening: HSIC, KSG MI.
- Portfolio and risk: tail dependence on top of a global measure.
- Rule of thumb: never ship a dependence claim built on Pearson alone.

## Reproducing this

```bash
git clone https://github.com/kovashikawa/correlation-models
cd correlation-models
uv venv && uv pip install -r requirements.txt
uv run python scripts/benchmark.py
```

The repo also includes a self-test that checks Chatterjee's xi against the canonical XICOR formulation, distance correlation against known cases, and tail dependence against closed-form values for Y = X and Y = |X|.

## Further reading

1. Szekely, Rizzo, Bakirov (2007). [Measuring and testing dependence by correlation of distances](https://projecteuclid.org/journals/annals-of-statistics/volume-35/issue-6/Measuring-and-testing-dependence-by-correlation-of-distances/10.1214/009053607000000505.full). *Annals of Statistics*.
2. Szekely and Rizzo (2009). [Brownian distance covariance](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-3/issue-4/Brownian-distance-covariance/10.1214/09-AOAS312.full). *Annals of Applied Statistics*.
3. Chatterjee (2021). [A new coefficient of correlation (PDF)](https://arxiv.org/pdf/1909.10140). *JASA*.
4. Gretton et al. (2005). [Measuring statistical dependence with Hilbert-Schmidt norms (PDF)](http://alex.smola.org/papers/2005/GreBouSmoSch05.pdf). *ALT*.
5. Kraskov, Stogbauer, Grassberger (2004). [Estimating mutual information (PDF)](https://arxiv.org/pdf/cond-mat/0305641). *Physical Review E*.
6. Reshef et al. (2011). [Detecting novel associations in large data sets](https://www.science.org/doi/10.1126/science.1205438). *Science*.
7. Kinney and Atwal (2014). [Equitability, mutual information, and the maximal information coefficient](https://www.pnas.org/doi/10.1073/pnas.1309933111). *PNAS*.
8. Joe (1997). [Multivariate models and dependence concepts](https://doi.org/10.1201/b13150). Chapman and Hall.
9. Song et al. (2012). [Feature selection via dependence maximization (PDF)](https://www.jmlr.org/papers/volume13/song12a/song12a.pdf). *JMLR*.
