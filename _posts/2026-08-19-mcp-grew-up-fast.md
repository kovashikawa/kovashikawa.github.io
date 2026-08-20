---
layout: single
title: "MCP Grew Up Fast: From Experiment to Enterprise Trust Boundary"
date: 2026-08-19 12:00:00 -0400
excerpt: "how an 18-month standards effort quietly turned into the compliance layer for AI-to-CRM access"
tags: [MCP, OAuth, Salesforce, Protocols, AI]
categories: [ai]
author_profile: true
toc: true
toc_sticky: true
read_time: true
---

## "Why should you ever log into Salesforce again?"

Salesforce co-founder Parker Harris asked that out loud this spring, opening a product announcement most companies would never risk making about their own flagship product. By April 2026, Salesforce had shipped the part that made the question serious: a hosted, MCP connection any AI client can use to query a CRM in plain English, generally available to every Enterprise Edition org.

I spent a the past few weeks building against exactly that surface. What struck me wasn't the demo. It was realizing how recently the plumbing underneath it didn't exist at all, and how fast it got built.

## The 18 months that made this possible

MCP is barely two years old. Anthropic open-sourced it on November 25, 2024, after two engineers, David Soria Parra and Justin Spahr-Summers, had been building it since that July. What happened between then and Salesforce's GA announcement is a compressed history of a protocol earning enterprise trust in real time, one hard problem at a time.

**Can two systems even speak the same language?** (Nov 2024 - Mar 2025)
MCP launched as a fairly narrow tool: JSON-RPC over stdio, mostly local processes talking to Claude Desktop. Useful for developers, not yet something a SaaS vendor would expose to the internet. That changed on March 26, 2025, when the spec's second version added Streamable HTTP transport and, for the first time, a real OAuth 2.1-based authorization framework. The same day, OpenAI publicly committed to supporting MCP across its products, turning it from "Anthropic's protocol" into the industry's protocol.

**Can you trust what's on the other end?** (Apr - Jun 2025)
Standardizing the wire format immediately exposed how little the security model had been stress-tested. In April 2025, Invariant Labs published a reproducible "tool poisoning attack," the first serious public MCP exploit. It forced the issue. By June 18, the spec formally classified every MCP server as an OAuth 2.1 resource server under RFC 9728 (Protected Resource Metadata), meaning a server now had a standard, spec-defined way to declare who it trusted and what it would accept. Five days later, Salesforce anchored its Agentforce 3 platform around MCP interoperability and shipped its first servers.

**Can an enterprise actually govern this?** (Jun - Nov 2025)
A resource-server model is necessary but not sufficient for a Fortune 500 security review. Through the summer, the ecosystem built the governance layer the spec hadn't yet: Cloudflare shipped MCP server portals, Auth0 published patterns for MCP-as-OAuth-resource-server, New Relic added MCP traffic observability. Then the November 25, 2025 spec, released on MCP's first anniversary, formalized a lot of that community work directly into the standard: Client ID Metadata Documents replaced ad hoc dynamic client registration, an "Enterprise-Managed Authorization" extension let a company's own identity provider broker trust instead of each vendor inventing its own, and PKCE went from recommended to mandatory.

**Is this now just infrastructure?** (Dec 2025 - Apr 2026)
Two weeks after that spec, on December 9, 2025, Anthropic donated MCP to the Linux Foundation's new Agentic AI Foundation, co-founded with Block and OpenAI, backed by Google, Microsoft, AWS, Cloudflare, and Bloomberg. A protocol that started as one company's open-source experiment was now nobody's proprietary asset. Four months later, Salesforce's Hosted MCP Servers went GA: OAuth 2.0 per user, scoped through an External Client App requesting `mcp_api` and `refresh_token` grants, fully managed on Salesforce's own infrastructure. Seventeen months, roughly, from a two-person side project to the default way an AI agent gets audited, revocable, read-only access to a live enterprise CRM.

## What that actually bought the people building on top of it

None of the above is abstract if you were the one wiring a client into it. Here's the honest counterfactual, without walking through what I actually built: two years ago, an integration like this meant inventing your own answer, from scratch, to "how do I prove this credential can only read, only as this user, only against this one resource," and then re-litigating that answer with every enterprise security team that asked. There was no shared vocabulary for it. Every vendor's OAuth implementation was a slightly different bespoke argument.

What the spec's maturation actually hands you now is that argument, pre-made. Audience-bound tokens are a defined behavior (RFC 8707), not a design choice you have to justify. Protected Resource Metadata means a server can advertise its own trust boundary instead of you documenting it out-of-band. Mandatory PKCE and a standardized insufficient-scope error path mean the failure modes a security reviewer asks about already have a spec-sanctioned answer. You're not inventing a compliance story anymore. You're citing one that a hundred other implementers already stress-tested.

That is, genuinely, the difference between a multi-quarter security review and something a small team can ship, get audited, and put in front of an enterprise admin in a reasonable window. Not because the engineering got easier. Because the trust model stopped being something each of us had to reinvent alone.

## A personal note

I didn't build any of the above. I'm one of a lot of engineers who happened to be building an integration during the window this protocol matured, and I got to feel the difference directly: a project that would have meant inventing a security model from scratch a year earlier instead meant adopting one that had already been through a public tool-poisoning exploit, a year of enterprise scrutiny, and formal standardization. That's not a personal achievement. It's what happens when a whole community, not just one vendor, spends eighteen months arguing about the right way to do this.

I'll be talking about a related piece of this, deterministic evaluation and typed tool contracts for AI agents, at Google Search Central Live this year. Same underlying point: the hard problem in this space isn't connecting to the data anymore. It's trusting what comes back, and that's only tractable now because the layer underneath got serious.

## Two years, not a decade

A two-person open-source project became a Linux Foundation standard with mandatory security guarantees, adopted by every major AI lab, in under two years. That's not normally how standards get made. Usually it takes a decade of committees. This one got there because a large number of people kept finding the gaps and fixing them in public, fast, under real pressure. Worth remembering next time an integration "just works" and it's tempting to forget how much had to go right first.
