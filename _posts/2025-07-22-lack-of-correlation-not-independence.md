---
title:  "Why Lack of Correlation Doesn’t Mean Independence"
excerpt: "Pearson’s ρ drops to zero for X vs |X|—a neat example of hidden dependence."
author: rafael            # matches whatever you have in _config.yml > authors
categories: [statistics]
tags: [correlation, independence, statistics, python]
canonical_url: https://medium.com/@rafaelkovashikawa/why-the-lack-of-correlation-doesnt-mean-independence-9506a9eec376
toc: true       
---

## The Intuition

For a **Gaussian** (Normal) pair of random variables, Pearson’s correlation coefficient ($\rho$) fully captures dependence. Step outside the Gaussian world, though, and $\rho$ can fail spectacularly.

Before diving in, recall the classical definition:

$$
\rho_{X,Y}=\frac{\operatorname{cov}(X,Y)}{\sigma_X\sigma_Y},
$$

where $\sigma$ denotes the standard deviation. Values are bound in the range $[-1, 1]$. A value of 0 is *often* mistaken for “independence” — but it only guarantees **no linear** relationship.

---

## A Simple Counter-Example

Let's consider two random variables:

- $X \sim \mathcal{N}(0,1)$
- $Y = \lvert X \rvert$

Intuitively, $Y$ is *completely determined* by $X$, so they are **dependent**. Let’s simulate this and see what Pearson’s $\rho$ says.

```python
import numpy as np
import pandas as pd

n = 100_000
x = np.random.normal(0, 1, n)
df = pd.DataFrame({"X": x, "Y": np.abs(x)})

print(df.corr())
```

The expected output will be something like this (values will fluctuate slightly):

|      | X        | Y        |
|------|----------|----------|
| **X** | 1.000000 | 0.000??? |
| **Y** | 0.000??? | 1.000000 |

As you can see, $\rho$ is approximately $0$! Pearson’s correlation fails here because the dependence is non-linear and symmetric around zero.

---

## Takeaways

* Zero correlation does not imply independence (unless your variables are jointly Gaussian).
* Always visualize your data or use a non-linear measure of correlation (e.g., Spearman, Kendall, distance correlation, mutual information).
* In risk analysis or alpha research, relying solely on Pearson's correlation may hide important tail dependencies or regime shifts.
