---
title: "Lowercase Is a Mood"
excerpt: "One bare a/A button in the masthead renders the whole site in lowercase. Display-only, so the source and the SEO stay proper case. A short note on why and how."
date: 2026-08-28
categories:
  - design
  - projects
tags:
  - css
  - typography
  - minimalism
---

The whole site is one click away from lowercase. There is a bare `a/A` button
in the masthead, next to Home. Click it and every heading, title, date and
excerpt renders lowercase. Click again and it goes back. The active letter is
bold, so the button says which mode you are in.

<video controls loop muted playsinline poster="/assets/videos/lc-lowercase-poster.png" width="100%">
  <source src="/assets/videos/lc-lowercase.mp4" type="video/mp4">
</video>

## Display-only

The feature is a single CSS rule applied to the body when the toggle is on:

```css
body.lc, body.lc * {
  text-transform: lowercase !important;
}
```

That is the whole thing. It is a rendering choice, not a content change. The
source, the front matter, the page titles and the RSS feed all stay proper
case. Search engines read the unmodified markup. So this is free: a purely
visual preference that costs nothing in legibility of the underlying content,
and nothing in SEO.

## Why bother

Lowercase reads quieter. All-caps carries weight and formality; lowercase
drops the volume. For a personal site that is mostly short notes and links,
that is the right register. It is also a small joke: a site whose whole
point is few words, styled to say them as softly as possible.

The state persists in `localStorage`, so the mood survives reloads.

## One button, two letters

The button itself is just `a` and `A` with a slash. No box, no border, no
icon. The active letter gets weight. It is the smallest control that can
state a binary choice about typography, which is exactly what it is.
