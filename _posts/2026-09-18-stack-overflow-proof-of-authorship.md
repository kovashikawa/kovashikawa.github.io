---
title: "Stack Overflow Is Selling Proof That a Human Wrote It"
excerpt: "Developer Story is back, sold as a verification layer for authorship. On the empty ledger, the notary framing, and who gets to certify a developer."
date: 2026-09-18
categories:
  - ai
tags:
  - stack-overflow
  - verification
  - identity
  - agents
---

Stack Overflow brought Developer Story back on September 10, four years after the
2022 shutdown. [The announcement](https://stackoverflow.blog/2026/09/10/re-introducing-developer-story/)
is unusually blunt about why.

The short version is that people don't need to come to the site for answers
anymore, "the chatbots have that covered," and most LLMs were trained on their
datasets. So the goal is not winning back question traffic. It is evidence that a
specific human wrote a specific thing.

The umbrella name is Stack Identity, described in the post as "the verified
developer proof-of-work layer to the internet." Day one ships "specialties,"
areas of proven expertise derived from tag activity, with a settings pane for
choosing which ones are displayed. Later comes material pulled from
"integrations with other sites that have verified data about you."

## Not a LinkedIn pivot

My first read was that this is Stack Overflow turning into LinkedIn. It isn't.
LinkedIn's moat is the network graph plus recruiter demand, and Stack Overflow
has neither and is not building either. This reads more like a notary. The
product is the signature, and the profile is just the surface it gets printed
on.

The 2022 sunset deleted the old Dev Story data for privacy compliance, so nobody
carried anything over. A notary with no back catalog has to sell a signature that
starts today. My own record is the illustration. Six years of membership, four
answers, one accepted, and the accepted one is the zero-score answer from 2023.
The answer that carries the number is a pandas reply from 2021 at score 122,
never accepted, second of twenty on the question. That is the whole verified
footprint, and none of it describes what I work on now.

## Verification needs a second party

A signature only has value when someone else accepts it. "Integrations with
other sites that have verified data about you" is carrying the entire
distribution argument, and no partner names are attached to it. The hard part is
not the profile UI or the trust model. It is getting an employer, a client, or
another platform to treat Stack Overflow as the identity authority.

They also run [agents.stackoverflow.com](https://agents.stackoverflow.com/), a
separate site in beta where the users are AI agents. A post there carries no
trust score at all until at least one independent verification reports that the
guidance worked in practice. Votes are read-time judgments, verifications are
use-time outcomes, and reputation only moves when other agents find the
contribution useful. The strictest proof standard in the company's portfolio is
pointed at machines, while the human product ships with tag activity and a
promise.

Both are bids to define what counts as proof of work. The agent side requires an
independent verification before it will score anything. The human side starts
with the tags you already answered.
