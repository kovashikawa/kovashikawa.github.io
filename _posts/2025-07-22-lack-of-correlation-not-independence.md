---
title: "Zero Correlation Does Not Imply Independence"
excerpt: "A zero correlation only rules out a linear link. See why Y = &#124;X&#124; defies Pearson's rho, and which measures catch the hidden dependence."
description: "A zero correlation only rules out a linear link. See why Y = &#124;X&#124; defies Pearson's rho, and which measures catch the hidden dependence."
author: rafael            # matches whatever you have in _config.yml > authors
categories: [statistics]
tags: [correlation, independence, statistics, python]
canonical_url: https://medium.com/@rafaelkovashikawa/why-the-lack-of-correlation-doesnt-mean-independence-9506a9eec376
toc: true
mathjax: true
---

> Zero correlation on its own does not prove independence; it only rules out a linear relationship. Dependence can hide in nonlinear patterns such as Y = \|X\|, where Pearson's rho equals zero yet X fully determines Y.

## The Intuition

For a **jointly Gaussian** pair of random variables, Pearson's correlation coefficient ($\rho$) fully captures dependence. Gaussian margins alone are not enough: a marginally normal pair can be uncorrelated yet fully dependent. Step outside the jointly Gaussian world, though, and $\rho$ can miss the dependence entirely.

Before diving in, recall the classical definition:

$$
\rho_{X,Y}=\frac{\operatorname{cov}(X,Y)}{\sigma_X\sigma_Y},
$$

where $\sigma$ denotes the standard deviation. Values are bounded in the range $[-1, 1]$. A value of 0 is *often* mistaken for "independence", but it only guarantees **no linear** relationship.

---

## A Simple Counter-Example

Let's consider two random variables:

- $X \sim \mathcal{N}(0,1)$
- $Y = \lvert X \rvert$

Intuitively, $Y$ is *completely determined* by $X$, so they are **dependent**. Let's simulate this and see what Pearson's $\rho$ says.

```python
import numpy as np
import pandas as pd

n = 100_000
rng = np.random.default_rng(42)
x = rng.normal(0, 1, n)
df = pd.DataFrame({"X": x, "Y": np.abs(x)})

print(df.corr())
```

The expected output (seeded, so reproducible) is:

|      | X        | Y        |
|------|----------|----------|
| **X** | 1.000000 | -0.001842 |
| **Y** | -0.001842 | 1.000000 |

As you can see, $\rho$ is approximately $0$! Pearson's correlation fails here because the dependence is non-linear and symmetric around zero. To see why $\rho$ is exactly 0 in the population: $\operatorname{cov}(X, \lvert X \rvert) = E[X\lvert X \rvert] - E[X]E[\lvert X \rvert]$, and both terms vanish for any distribution symmetric about zero with finite variance. The sample value is small but not exactly 0, since the sample standard error of $r$ is about $\sqrt{3/n} \approx 0.0055$ at this sample size because $Y$ is a function of $X$.

---

## Takeaways

* Zero correlation does not imply independence (unless your variables are jointly Gaussian).
* Plot the pair when you can. For formal analysis, use a measure of dependence that is not limited to linear or monotone relationships, such as distance correlation, mutual information, or Chatterjee's xi. Note that rank measures (Spearman, Kendall) also report zero on $Y = \lvert X \rvert$, since they capture monotone association only.
* In risk analysis or alpha research, relying solely on Pearson's correlation may hide nonlinear and tail dependence.
