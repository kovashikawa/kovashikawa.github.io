---
title: "A LinkedIn Cover That Isn't Stock"
excerpt: "I replaced a stock gradient banner with one line of math. The equation is geometric Brownian motion; the curve is a single realized path; the whole thing is grey because a cover should be ambient. Along the way: why PNG from HTML comes out soft, and the 3x-then-downscale fix."
date: 2026-08-22
categories:
  - projects
  - design
tags:
  - linkedin
  - mathjax
  - visualization
  - brand
---

My LinkedIn cover used to be a purple gradient with the company's tagline on
it. It works for the company page. My profile is my own surface, so I wanted
a cover that read as mine: a quiet, monochrome plot in the same palette as
this blog, with no words trying to sell anything.

The final result is one line of math and one curve:

![The cover](/assets/images/li-cover.png)

The equation is geometric Brownian motion,

$$ dX_t = \mu X_t\,dt + \sigma X_t\,dW_t $$

the process Black-Scholes assumes for the underlying. The curve is a single
realized path: exponential drift, Brownian noise. Same idea as the
volatility smile page I wrote earlier. The elegant model, and what actually
happens when you let randomness in.

A few decisions along the way were deliberate.

## A cover should be ambient

A LinkedIn banner is background. People come to the profile for the bio, the
experience, the work. So the cover gets three elements and nothing else: the
curve, a caption, and a URL. No axes, no ticks, no gridlines inside the plot.
If the values don't matter, the scaffolding is noise. I kept the graph-paper
grid from this blog's background, because it gives the curve a sense of place
without making it a chart.

## Grey, not black, not blue

I spent a while on color before landing on none. The data-viz literature
converges on "grey plus one accent" for figures where a specific element
matters. But a cover is the opposite situation: nothing in it should compete
with the content below it. So the curve and caption are a mid grey, the URL
is a little darker for legibility, and there is zero hue. Monochrome is the
look this blog already has; the cover just commits to it harder.

## Text on a cover dies

My first versions had the equation floating in the upper band, then as a
caption at the bottom, then on the left. The placement that survived is
symmetric: URL top-right, equation bottom-right, both right-aligned to the
same column. An equation is content that describes the visual, so it sits on
the visual like a caption. The URL is a signature, so it sits in the corner
like one. Two text elements, two jobs, no overlap.

## The crispness problem

The annoying part was output quality. Screenshot the HTML at 1x and a 2px
curve is 2 physical pixels. It comes out soft and aliased, and LinkedIn's
own compression makes it worse. The fix is a pipeline I now use every time I
turn HTML into an image:

1. Render in headless Chrome at 3x with `--force-device-scale-factor=3`,
   giving a 4752x1188 master where every line and glyph has real
   antialiasing data.
2. Downscale to the exact target with Pillow's LANCZOS resampler.
   `sips -z` is bilinear and softens edges; LANCZOS is the right filter.
3. Save as PNG, or JPEG around q=90 for a much smaller upload.

The text needed to be a touch bigger than I first wanted (17px, not 15px).
At 1584px wide, small glyphs are at the legibility floor, and shrinking
them to fit the canvas just makes them blur.

## The tool

All of it is one Python file now, so the next cover is one command:

```bash
kcover --seed 42 --out cover.png
```

It simulates a fresh path each run. Same equation, different realization,
which is the point. The seed pins a specific draw if you want to keep one.
It renders, downscales, and verifies the curve doesn't collide with the
equation or the URL.

Repo: [github.com/kovashikawa/kcover](https://github.com/kovashikawa/kcover)

The cover is live on [my profile](https://www.linkedin.com/in/rkovashikawa/).
If you see it in the wild, the curve you're looking at is one sample path of
a stochastic differential equation. That's the whole joke.
