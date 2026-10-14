---
layout: single
title: "Lowercase Is a Mood, Now a Chrome Extension"
date: 2026-10-14 12:00:00 -0400
excerpt: "The a/A switch from this site, packaged as a Chrome extension that renders any site in lowercase. Display-only, saved per site, with a demo on Wikipedia."
description: "A Chrome extension with one a/A switch that renders the site you are on in lowercase. Display-only, saved per site, no network requests."
categories:
  - design
  - projects
tags:
  - chrome-extension
  - css
  - typography
  - minimalism
author_profile: true
---

This site has an `a/A` button that renders every page in lowercase. I liked it enough to want it everywhere, so it is now a free Chrome extension: [Lowercase Is a Mood](https://chromewebstore.google.com/detail/lowercase-is-a-mood/pdbkpobppnhekkcnahnabccppcbmahbe).

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;">
  <iframe src="https://www.youtube-nocookie.com/embed/GwMf4hCyZHw" title="Lowercase Is a Mood, a Chrome extension that renders any site in lowercase" style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;" allow="encrypted-media; picture-in-picture; fullscreen" allowfullscreen loading="lazy"></iframe>
</div>

[Watch on YouTube](https://youtu.be/GwMf4hCyZHw) if the player does not load.

## What it does

Click the extension's toolbar icon and press `a/A`. The bold letter is the active mode. The current site renders in lowercase, and a second click brings back the original case.

The choice is saved per site, so a site you switched stays lowercase on reload and in new tabs, and other sites are untouched.

## The demo

The video is one screen recording with no cuts or speed-ups. It starts on Wikipedia's [Longest pages](https://en.wikipedia.org/wiki/Special:LongPages) list, switches it to lowercase, and opens the longest article on the list. Then it goes back and opens another long one. Both articles open already in lowercase, because the setting belongs to the site rather than the page.

## Display-only

The extension does not rewrite page text, form values or page source. It adds one class to the page's root element, and a single CSS rule on that class sets `text-transform: lowercase`. It runs at document start, so a site you chose to see in lowercase never flashes in its original case.

It collects nothing. There are no network requests, analytics or accounts, and the only thing stored is one on/off value per site in Chrome's local storage. The [code is public](https://github.com/kovashikawa/lowercase-is-a-mood-extension), and each release is built in public CI with a signed provenance attestation.

## Limits

It lowercases everything on the page, including code and acronyms. Chrome does not let extensions change its internal pages or the Web Store, and some embedded or shadow DOM content cannot be changed. Pages that were open before you installed it need one reload.

For the original version of the idea, see [Lowercase Is a Mood]({% post_url 2026-08-28-lowercase-is-a-mood %}), the note on this site's own button.
