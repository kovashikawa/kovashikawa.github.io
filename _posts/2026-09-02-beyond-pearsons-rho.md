---
layout: single
title: "Beyond Correlation: measuring dependence the modern way"
date: 2026-09-02 12:00:00 -0400
excerpt: "Pearson says 0, but X and Y are dependent. A guide to distance correlation, Chatterjee's xi, HSIC, KSG MI, and tail dependence, with a rerunnable benchmark."
description: "A guide to distance correlation, Chatterjee's xi, HSIC, KSG mutual information, and tail dependence: what each measure catches that Pearson misses, benchmarked against nine synthetic datasets."
tags: [statistics, correlation, dependence, python, benchmark]
categories: [statistics]
author_profile: true
toc: true
toc_sticky: true
read_time: true
mathjax: true
---

Pearson's correlation is zero for plenty of pairs that are fully dependent. The canonical case: $X \sim \mathcal{N}(0,1)$, $Y = \lvert X \rvert$. $\rho$ reports about 0, but $X$ determines $Y$ exactly. The fix isn't "use a nonlinear measure" in the abstract, it's knowing which measure catches which failure mode, and what each one actually guarantees.

That's what this post benchmarks: distance correlation, Chatterjee's xi, HSIC, KSG mutual information, and tail dependence, against nine synthetic datasets built to break Pearson in different ways.

Every number below is reproducible: seeded, and the companion repo is [kovashikawa/correlation-models](https://github.com/kovashikawa/correlation-models).

Two minutes covering the same ground visually, walking through each counterexample and measure below.

<video controls muted playsinline poster="/assets/videos/beyond-correlation-poster.png" width="100%">
  <source src="/assets/videos/beyond-correlation.mp4" type="video/mp4">
</video>

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

<p style="font-size: 0.75em; color: var(--muted, #555555);" markdown="1">n = 10,000, seed 42, full generator code in the repo.</p>

The first three rows are the classical toolkit, and none of them catch quadratic, abs, sine, circle, or cross. Chatterjee's xi, distance correlation, and KSG MI catch all five. HSIC and tail dependence are subtler cases, covered below. That gap is the whole problem, quantified.

Two things worth flagging before the per-measure sections.

1. **Cross is the sharpest counterexample.** $Y = X \cdot W$, where $W$ is a random sign independent of $X$. The sign cancels out any linear relationship, so Pearson, Spearman, and Kendall all read near zero. But $Y$ is not independent of $X$: $\lvert Y \rvert = \lvert X \rvert$ always. Chatterjee's xi (0.245) and distance correlation (0.313) both catch it. Both margins are standard normal here, and the pair still isn't jointly Gaussian, which is exactly the case where "uncorrelated implies independent" fails.
2. **The measures split into two questions.** Xi, distance correlation, HSIC, and KSG MI test dependence across the whole distribution. Tail dependence tests something narrower: whether extremes move together, which is why the tail_t column exists. HSIC is unnormalized and MI is measured in nats, so don't compare magnitudes across measures. Use them to detect and rank, not to score.
3. **The independent column is the null.** Every measure should read near zero there, and does: Chatterjee's xi is -0.004, distance correlation 0.014, HSIC 0.000. That's the reference for judging every "catch" above. Xi's null standard deviation is about $\sqrt{2/(5n)} \approx 0.0063$ at $n = 10{,}000$ (derived below), which puts circle's xi of 0.254 roughly 40 null standard deviations out, not just "nonzero."

## Distance correlation

Szekely, Rizzo and Bakirov (2007) introduced distance correlation to fix exactly this blind spot. The idea: independence is equivalent to the joint characteristic function factoring into the product of marginals. Distance covariance is a weighted norm on exactly that difference, and the estimator falls out as a double-centering of pairwise distance matrices:

$$\operatorname{dCor}(X,Y) = \frac{\operatorname{dCov}(X,Y)}{\sqrt{\operatorname{dCov}(X,X)\,\operatorname{dCov}(Y,Y)}}.$$

The property that matters: $\operatorname{dCor} = 0$ if and only if $X$ and $Y$ are independent, for distributions with finite first moments, in any dimension. Pearson cannot make that claim. In the bivariate normal case dCor is a deterministic function of $\lvert \rho \rvert$ and never exceeds it.

Cost: the naive estimator used here is $O(n^2)$ memory and time, because of the pairwise distance matrices. Fine at 10k rows, painful at 10M. Faster $O(n \log n)$ algorithms exist for univariate data (Huo and Szekely 2016).

## Chatterjee's rank correlation

Chatterjee (2021) took a different route, with a coefficient that is almost absurdly simple. Rank X, reorder Y's max-ranks by X, and measure how much adjacent ranks jump:

$$\xi_n(X,Y) = 1 - \frac{A_1}{C_U}, \qquad A_1 = \frac{1}{2n}\sum_{i=1}^{n-1}\left|\frac{r_{i+1}}{n} - \frac{r_i}{n}\right|, \qquad C_U = \frac{1}{n}\sum_{i=1}^{n} g_i(1-g_i),$$

where $r_i$ are the max-ranks of $Y$ reordered by $X$ and $g_i = (\text{max-rank of } -Y_i)/n$, so that $r_i/n$ and $g_i$ lie in $[0, 1]$. With no ties this collapses to $\xi = 1 - \frac{3}{n^2-1}\sum_{i=1}^{n-1}\lvert r_{i+1} - r_i \rvert$. For non-constant $Y$, the population coefficient $\xi$ satisfies: $\xi = 0$ iff independence, $\xi = 1$ iff $Y$ is a measurable function of $X$. Computes in $O(n \log n)$ and is completely nonparametric.

Three footnotes. First, under independence $\xi_n$ has mean zero and standard deviation about $\sqrt{2/(5n)}$, so it lands negative about half the time; that is expected, not a bug. Second, $\xi(X, Y)$ is asymmetric by construction: it measures "how well Y behaves as a function of X," which the paper argues for deliberately. Third, xi has low power against many smooth alternatives (Shi, Drton and Han 2022), so treat a low xi as "no strong signal," not "no dependence."

## HSIC

HSIC comes from the kernel methods literature (Gretton et al. 2005) and became a standard tool in nonlinear feature selection (Song et al. 2012). Map each variable into a reproducing kernel Hilbert space with a universal kernel (RBF here), and take the squared Hilbert-Schmidt norm of the cross-covariance operator between the two embeddings. HSIC = 0 iff independence for universal kernels (Gretton 2005 on compact domains; Fukumizu et al. 2008 for the general statement).

On this benchmark HSIC is the flattest row: 0.088 on linear, dropping to 0.008 on sine and 0.000 on independent, everything within a factor of about 11. It still separates dependent from independent, linear and heavy_tail sit well above the 0.000 null, but the RBF kernel's median-distance bandwidth is not tuned per dataset here, so treat the exact HSIC values as a coarse detector rather than a ranking. Its real strength is scaling to vector-valued or high-dimensional X and Y, where density-based measures like KSG MI become impractical.

## KSG mutual information

Mutual information $I(X; Y) = 0$ iff independence, full stop. The KSG estimator (Kraskov, Stogbauer and Grassberger 2004) is a k-nearest-neighbor scheme that adapts its resolution in both margins, which fixes the classic histogram-bin problems. It is what scikit-learn's `mutual_info_regression` uses under the hood, with $k = 3$ neighbors here.

The values are not comparable across datasets. Linear has additive noise and a finite population MI of $-\frac{1}{2}\ln(1-\rho^2)$. At the generator's $\rho \approx 0.9806$ that works out to about 1.629 nats, matching the 1.639 estimate. Quadratic, abs, sine, circle, and cross have no additive noise: each Y is fully determined by X and a coin flip or angle draw with no residual randomness, so the joint distribution is singular and the population MI is infinite. The KSG estimate for these just grows with $n$ and shrinks with $k$. So "6.4 nats on abs" does not mean abs is four times more dependent than linear's 1.6. MI answers "is there dependence" decisively, and "how strong" only loosely.

The maximal information coefficient (Reshef et al. 2011) is the other mutual-information-based measure people reach for, maximizing normalized MI over grid binning schemes. It made a splash in *Science* in 2011, but its equitability claims were shown mathematically impossible for any nontrivial measure (Kinney and Atwal 2014), and later power comparisons found it underpowered relative to distance correlation (Simon and Tibshirani 2014). Refined variants followed (Reshef et al. 2016). Treat MIC, like the measures above, as a detector rather than a calibrated strength scale. It is not in this benchmark: `minepy`, the standard implementation, requires a compiled extension and was left out; the Reshef lab's Java MINE tool is the reference implementation.

## Tail dependence

The measures above test dependence across the whole distribution. Risk work cares about the tails specifically: given that one asset is above its 95th percentile, how likely is the other to be too? Tail dependence coefficients go back to Sibuya (1960); the modern treatment is Joe (1997).

The population quantity is the limit, if it exists:

$$\lambda_U = \lim_{q \to 1} P(F_Y(Y) > q \mid F_X(X) > q).$$

The benchmark row reports the finite-quantile estimator $\lambda(q)$ at $q = 0.95$, a standard VaR level, exactly 500 conditioning exceedances at $n = 10{,}000$. Under independence, $\lambda(q) = 1 - q = 0.05$ exactly. The independent column reads 0.032, about 1.8 standard errors low (SE $\approx \sqrt{0.05 \cdot 0.95 / 500} \approx 0.0098$), consistent with the null.

This is where the Gaussian copula earns its infamy. For any correlation $\rho < 1$, the Gaussian copula has zero tail dependence in the limit, but the co-exceedance probability vanishes slowly as $q \to 1$. The linear column is generated from a Gaussian copula at $\rho \approx 0.9806$. Its finite-q estimator reads 0.856 at $q = 0.95$ against a closed form of 0.838. That gap narrows only gradually: about 0.74 by $q = 0.999$ and 0.70 by $q = 0.9999$. If your risk model is Gaussian-copula shaped and you feed it Pearson correlations, you are asserting away joint tail risk by construction, just with a delay.

The tail_t column is the counterexample. It's generated from a $t$-copula with $\nu = 3, \rho = 0.5$, which has positive asymptotic tail dependence: $\lambda_U = 2T_4(-\sqrt{4/3}) \approx 0.31$. The empirical estimate at $q = 0.95$ reads 0.376, higher than the asymptotic limit because finite-q estimates overstate $\lambda_U$ and converge to it slowly, the same effect visible in the linear column above.

The heavy_tail column is a warning about reading too much into a name. It's just $X$ plus independent $t_3$ noise, which is asymptotically tail independent despite the heavy-tailed marginal. Larger simulations ($n = 20$M) show its $\lambda(q)$ decaying with $q$: 0.487 at $q = 0.95$, 0.011 at $q = 0.999$, below 0.001 at $q = 0.9999$. The 0.500 in the benchmark table is the $n = 10{,}000$ estimate at $q = 0.95$ only, consistent with the larger simulation's first point; the decay only shows up once $q$ moves well past 0.95. Always check the generator, not the label.

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

The repo also includes a self-test that checks Chatterjee's xi against the canonical XICOR formulation, distance correlation against known cases, tail dependence against closed-form values for $Y = X$ and $Y = \lvert X \rvert$, and the t-copula column against its closed-form $\lambda_U$.

## Further reading

1. Szekely, Rizzo, Bakirov (2007). [Measuring and testing dependence by correlation of distances](https://projecteuclid.org/journals/annals-of-statistics/volume-35/issue-6/Measuring-and-testing-dependence-by-correlation-of-distances/10.1214/009053607000000505.full). *Annals of Statistics*.
2. Szekely and Rizzo (2009). [Brownian distance covariance](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-3/issue-4/Brownian-distance-covariance/10.1214/09-AOAS312.full). *Annals of Applied Statistics*.
3. Chatterjee (2021). [A new coefficient of correlation (PDF)](https://arxiv.org/pdf/1909.10140). *JASA*.
4. Gretton et al. (2005). [Measuring statistical dependence with Hilbert-Schmidt norms (PDF)](http://alex.smola.org/papers/2005/GreBouSmoSch05.pdf). *ALT*.
5. Kraskov, Stogbauer, Grassberger (2004). [Estimating mutual information (PDF)](https://arxiv.org/pdf/cond-mat/0305641). *Physical Review E*.
6. Reshef et al. (2011). [Detecting novel associations in large data sets](https://www.science.org/doi/10.1126/science.1205438). *Science*.
7. Kinney and Atwal (2014). [Equitability, mutual information, and the maximal information coefficient](https://www.pnas.org/doi/10.1073/pnas.1309933111). *PNAS*.
8. Sibuya (1960). Bivariate extreme statistics, I. *Annals of the Institute of Statistical Mathematics* 11(3):195-210.
9. Joe (1997). [Multivariate models and dependence concepts](https://doi.org/10.1201/b13150). Chapman and Hall.
10. Song et al. (2012). [Feature selection via dependence maximization (PDF)](https://www.jmlr.org/papers/volume13/song12a/song12a.pdf). *JMLR*.
11. Fukumizu, Gretton, Sun, Scholkopf (2008). [Kernel measures of conditional dependence (PDF)](https://papers.nips.cc/paper/3340-kernel-measures-of-conditional-dependence.pdf). *NIPS 20 (2008)*.
12. Huo and Szekely (2016). [Fast computing for distance covariance](https://www.tandfonline.com/doi/abs/10.1080/00401706.2015.1054435). *Technometrics*.
13. Shi, Drton, Han (2022). [On the power of Chatterjee's rank correlation (PDF)](https://arxiv.org/pdf/2008.06820). *Biometrika*.
14. Simon and Tibshirani (2014). [Comment on "Detecting novel associations in large data sets" by Reshef et al. (PDF)](https://arxiv.org/pdf/1401.7645). *arXiv:1401.7645*.
15. Reshef, Reshef, Mitzenmacher, Sabeti (2014). [Cleaning up the record on the maximal information coefficient and equitability](https://www.pnas.org/doi/10.1073/pnas.1408920111). *PNAS* 111(33):E3362-E3363.
16. Reshef, Reshef, Finucane, Sabeti, Mitzenmacher (2016). [Measuring dependence powerfully and equitably (PDF)](https://jmlr.csail.mit.edu/papers/volume17/15-308/15-308.pdf). *JMLR* 17(211):1-63.
