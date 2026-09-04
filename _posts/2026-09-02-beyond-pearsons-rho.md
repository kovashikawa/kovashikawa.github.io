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

[Last time]({% post_url 2025-07-22-lack-of-correlation-not-independence %}) I showed the canonical trap: $X \sim \mathcal{N}(0,1)$, $Y = \lvert X \rvert$, and Pearson's $\rho$ reports about 0 while $X$ fully determines $Y$. The one-line takeaway was "use a nonlinear measure." This post is the follow-through: which measures, what they actually guarantee, and what they miss.

Every number below comes from a runnable benchmark against eight synthetic datasets, seeded and reproducible. The companion repo is [kovashikawa/correlation-models](https://github.com/kovashikawa/correlation-models).

## The benchmark table

| Measure | linear | quadratic | abs | sine | circle | cross | independent | heavy_tail | tail_t |
|---|---|---|---|---|---|---|---|---|---|
| Pearson | 0.980 | -0.017 | -0.015 | 0.004 | 0.004 | -0.028 | 0.000 | 0.729 | 0.484 |
| Spearman | 0.978 | -0.014 | -0.014 | 0.026 | 0.003 | -0.010 | 0.004 | 0.795 | 0.477 |
| Kendall | 0.874 | -0.011 | -0.011 | 0.023 | 0.000 | -0.007 | 0.003 | 0.614 | 0.343 |
| Chatterjee xi | 0.812 | 0.999 | 0.999 | 0.997 | 0.254 | 0.245 | -0.004 | 0.449 | 0.164 |
| Distance corr | 0.971 | 0.543 | 0.559 | 0.253 | 0.197 | 0.313 | 0.014 | 0.768 | 0.468 |
| HSIC | 0.088 | 0.045 | 0.051 | 0.008 | 0.019 | 0.033 | 0.000 | 0.043 | 0.010 |
| KSG MI | 1.639 | 6.102 | 6.378 | 4.716 | 5.167 | 6.194 | 0.001 | 0.568 | 0.192 |
| Tail dep (q=0.95) | 0.856 | 0.498 | 0.498 | 0.090 | 0.000 | 0.462 | 0.032 | 0.500 | 0.376 |

All numbers: $n = 10{,}000$, seed 42, `scripts/benchmark.py`. Generators: linear = $1.5X + \mathcal{N}(0, 0.3^2)$, quadratic = $X^2$, abs = $\lvert X \rvert$, sine = $\sin(4X)$, circle = $(\cos\theta, \sin\theta)$ with $\theta \sim U(0, 2\pi)$, cross = $X \cdot W$ with $W$ a Rademacher sign, independent = two independent normals, heavy_tail = $X + 0.5 \cdot t_3$, tail_t = a $t$-copula with $\nu = 3, \rho = 0.5$ via the canonical construction.

The first three rows are the classical toolkit, and they all say "no dependence" on quadratic, abs, sine, circle, and cross. Chatterjee xi, distance correlation, and KSG MI flag all of them as dependent; HSIC and the tail-dependence column are subtler, see below. That contrast is the whole problem, quantified.

Two things to notice before the details:

1. **The cross column is the cleanest counterexample.** Cross is $Y = X \cdot W$ with $W$ a Rademacher sign. The sign is independent noise, so $Y$ is *not* a function of $X$: $\lvert Y \rvert = \lvert X \rvert$ is. That distinction is exactly why Chatterjee xi lands at 0.245, far below 1. Pearson, Spearman, and Kendall all report near zero because the sign symmetry cancels, while xi and distance correlation (0.313) flag the dependence. This is the canonical demonstration that zero correlation does not mean nothing is going on, and it doubles as a reminder that the "jointly Gaussian" qualifier is load-bearing: both margins here are standard normal.
2. **Global and local measures answer different questions.** Chatterjee xi, distance correlation, HSIC, and KSG MI measure dependence across the whole distribution. Tail dependence and the tail_t column measure co-exceedance in the extremes, a separate axis (see the tail section). HSIC is nonnegative and not normalized; MI is in nats. Use them as detectors and for ranking, not as comparable strengths.

## Distance correlation

Szekely, Rizzo and Bakirov (2007) introduced distance correlation to fix exactly this blind spot. The idea: independence is equivalent to the joint characteristic function factoring into the product of marginals. Distance covariance is a weighted norm on exactly that difference, and the estimator falls out as a double-centering of pairwise distance matrices:

$$\operatorname{dCor}(X,Y) = \frac{\operatorname{dCov}(X,Y)}{\sqrt{\operatorname{dCov}(X,X)\,\operatorname{dCov}(Y,Y)}}.$$

The property that matters: $\operatorname{dCor} = 0$ if and only if $X$ and $Y$ are independent, for distributions with finite first moments, in any dimension. Pearson cannot make that claim. In the bivariate normal case dCor is a deterministic function of $\lvert \rho \rvert$ and never exceeds it.

Cost: the naive estimator used here is $O(n^2)$ memory and time, because of the pairwise distance matrices. Fine at 10k rows, painful at 10M. Faster $O(n \log n)$ algorithms exist for univariate data (Huo and Szekely 2016).

## Chatterjee's xi

Chatterjee (2021) took a different route, with a coefficient that is almost absurdly simple. Rank X, reorder Y's max-ranks by X, and measure how much adjacent ranks jump:

$$\xi_n(X,Y) = 1 - \frac{A_1}{C_U}, \qquad A_1 = \frac{1}{2n}\sum_{i=1}^{n-1}\left|\frac{r_{i+1}}{n} - \frac{r_i}{n}\right|, \qquad C_U = \frac{1}{n}\sum_{i=1}^{n} g_i(1-g_i),$$

where $r_i$ are the max-ranks of $Y$ reordered by $X$ and $g_i = (\text{max-rank of } -Y_i)/n$, both normalized by $n$. With no ties this collapses to $\xi = 1 - \frac{3}{n^2-1}\sum_{i=1}^{n-1}\lvert r_{i+1} - r_i \rvert$. For non-constant $Y$, the population coefficient $\xi$ satisfies: $\xi = 0$ iff independence, $\xi = 1$ iff $Y$ is a measurable function of $X$. Computes in $O(n \log n)$ and is completely nonparametric.

Three honest footnotes. First, under independence $\xi_n$ has mean zero and standard deviation about $\sqrt{2/(5n)}$, so it lands negative about half the time; that is expected, not a bug. Second, $\xi(X, Y)$ is asymmetric by construction: it measures "how well Y behaves as a function of X," which the paper argues for deliberately. Third, xi has low power against many smooth alternatives (Shi, Drton and Han 2022), so treat a low xi as "no strong signal," not "no dependence."

## HSIC

HSIC comes from the kernel methods literature (Gretton et al. 2005) and became a standard tool in nonlinear feature selection (Song et al. 2012). Map each variable into a reproducing kernel Hilbert space with a universal kernel (RBF here), and take the squared Hilbert-Schmidt norm of the cross-covariance operator between the two embeddings. HSIC = 0 iff independence for universal kernels (Gretton 2005 on compact domains; Fukumizu et al. 2008 for the general statement), with no density estimation anywhere in the pipeline. That is what makes it practical in high dimensions, where density estimation is a nonstarter.

Bandwidth choice matters. The implementation in the companion repo uses the median-distance heuristic, which is the standard default.

## KSG mutual information

Mutual information $I(X; Y) = 0$ iff independence, full stop. The KSG estimator (Kraskov, Stogbauer and Grassberger 2004) is a k-nearest-neighbor scheme that adapts its resolution in both margins, which fixes the classic histogram-bin problems. It is what scikit-learn's `mutual_info_regression` uses under the hood.

The caveat: the values are not comparable across datasets. Linear has additive noise and a finite population MI of $-\frac{1}{2}\ln(1-\rho^2) = 1.64$ nats at $\rho = 0.98$, which the estimate matches. Quadratic, abs, sine, circle, and cross are noiseless functions: their population MI is infinite, and the KSG estimate just grows with $n$ (and shrinks with $k$). So "6.4 nats on abs" does not mean abs is "four times more dependent" than linear. MI answers "is there dependence" decisively, and "how strong" only loosely.

## MIC

The maximal information coefficient (Reshef et al. 2011) maximizes normalized mutual information over all grid binning schemes, capped by sample size. It made a splash in Science for "detecting novel associations in large data sets" and is the measure most people name when they want "the nonlinear correlation."

The footnote: the equitability claims were contested, with mathematical arguments showing the proposed definition of equitability is impossible for any nontrivial measure (Kinney and Atwal 2014), and independent power comparisons found MIC underpowered relative to distance correlation and other plug-in statistics (Simon and Tibshirani 2014); the original authors later replied with refined variants (Reshef et al. 2016). Treat MIC as one more detector, not a calibrated strength scale. `minepy` is a third-party implementation requiring a compiled extension, so it is left out of this repo's benchmark; the reference implementation is the Reshef lab's Java MINE tool.

## Tail dependence

All of the above measure dependence across the whole distribution. Risk work cares about the tails specifically: given that one asset is above its 95th percentile, how likely is the other to be too? Tail dependence coefficients go back to Sibuya (1960) and Joe (1997).

The population quantity is the limit, if it exists:

$$\lambda_U = \lim_{q \to 1} P(F_Y(Y) > q \mid F_X(X) > q).$$

The benchmark column reports the finite-quantile estimator $\lambda(q)$ at $q = 0.95$, a standard VaR level (roughly 500 conditioning exceedances at $n = 10{,}000$). Under independence, $\lambda(q) = 1 - q = 0.05$ exactly, so the independent column at 0.032 is sitting at the null floor.

This is where the Gaussian copula earns its infamy. For any correlation $\rho < 1$, the Gaussian copula has zero tail dependence in the limit: the co-exceedance probability vanishes as $q \to 1$, but slowly. At $\rho = 0.98$ the finite-q estimator still reads 0.856 at $q = 0.95$ (closed form 0.840), and only drifts down to about 0.7 by $q = 0.999$. If your risk model is Gaussian-copula shaped and you feed it Pearson correlations, you are asserting away joint tail risk by construction, just with a delay.

The tail_t column is the counterexample: a $t$-copula with $\nu = 3, \rho = 0.5$ has positive asymptotic tail dependence, $\lambda_U = 2T_4(-\sqrt{4/3}) \approx 0.31$, and the empirical column holds at 0.376. And the heavy_tail column is a warning about reading too much into a name: it is just $X$ plus independent $t_3$ noise, which is asymptotically tail independent. Its $\lambda(q)$ falls from 0.500 at $q = 0.95$ to near zero by $q = 0.9999$, so it is not a tail-dependent copula. Always check the generator, not the label.

## When to use what

- Bivariate, monotone, want a sign: Spearman or Kendall.
- Bivariate, any shape, want a [0, 1] strength: distance correlation or Chatterjee xi.
- High-dimensional or vector-valued: distance correlation, HSIC.
- Nonlinear feature screening: HSIC, KSG MI.
- Portfolio and risk: tail dependence on top of a global measure.
- Rule of thumb: never ship an *independence* claim, or a strength-of-relationship number, built on Pearson alone. A clearly nonzero Pearson correlation is itself sufficient evidence of dependence.

## Reproducing this

```bash
git clone https://github.com/kovashikawa/correlation-models
cd correlation-models
uv venv
uv pip install -e .
uv run python scripts/benchmark.py
```

The repo also includes a self-test that checks Chatterjee's xi against the canonical XICOR formulation, distance correlation against known cases, tail dependence against closed-form values for Y = X and Y = \|X\|, and the t-copula column against its closed-form $\lambda_U$.

## Further reading

1. Szekely, Rizzo, Bakirov (2007). [Measuring and testing dependence by correlation of distances](https://projecteuclid.org/journals/annals-of-statistics/volume-35/issue-6/Measuring-and-testing-dependence-by-correlation-of-distances/10.1214/009053607000000505.full). *Annals of Statistics*.
2. Szekely and Rizzo (2009). [Brownian distance covariance](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-3/issue-4/Brownian-distance-covariance/10.1214/09-AOAS312.full). *Annals of Applied Statistics*.
3. Chatterjee (2021). [A new coefficient of correlation (PDF)](https://arxiv.org/pdf/1909.10140). *JASA*.
4. Gretton et al. (2005). [Measuring statistical dependence with Hilbert-Schmidt norms (PDF)](http://alex.smola.org/papers/2005/GreBouSmoSch05.pdf). *ALT*.
5. Kraskov, Stogbauer, Grassberger (2004). [Estimating mutual information (PDF)](https://arxiv.org/pdf/cond-mat/0305641). *Physical Review E*.
6. Reshef et al. (2011). [Detecting novel associations in large data sets](https://www.science.org/doi/10.1126/science.1205438). *Science*.
7. Kinney and Atwal (2014). [Equitability, mutual information, and the maximal information coefficient](https://www.pnas.org/doi/10.1073/pnas.1309933111). *PNAS*.
8. Sibuya (1960). Bivariate extreme statistics. *Annals of the Institute of Statistical Mathematics* 11.
9. Joe (1997). [Multivariate models and dependence concepts](https://doi.org/10.1201/b13150). Chapman and Hall.
10. Song et al. (2012). [Feature selection via dependence maximization (PDF)](https://www.jmlr.org/papers/volume13/song12a/song12a.pdf). *JMLR*.
11. Fukumizu, Gretton, Sun, Scholkopf (2008). [Kernel measures of conditional dependence (PDF)](https://papers.nips.cc/paper/3340-kernel-measures-of-conditional-dependence.pdf). *NeurIPS*.
12. Huo and Szekely (2016). [Fast computing for distance covariance](https://www.tandfonline.com/doi/abs/10.1080/00401706.2015.1054435). *Technometrics*.
13. Shi, Drton, Han (2022). [On the power of Chatterjee's rank correlation (PDF)](https://arxiv.org/pdf/2008.06820). *Biometrika*.
14. Simon and Tibshirani (2014). [Comment on "Detecting novel associations in large data sets"](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-8/issue-1/Comment-on-Detecting-novel-associations-in-large-data-sets/10.1214/14-AOAS700A.full). *Annals of Applied Statistics*.
15. Reshef et al. (2016). [MINE: progressive disclosure of multivariate relationships in large data sets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0153744). *PLOS ONE*.
