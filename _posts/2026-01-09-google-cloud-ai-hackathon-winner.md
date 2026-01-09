---
title: "Winning the Google Cloud AI Hackathon: Building Multimodal Medical Annotation at Scale"
excerpt: "How we designed a production-oriented multimodal system for medical image annotation and won the December 2025 Google Cloud AI Hackathon."
date: 2026-01-09
last_modified_at: 2026-01-09
categories:
  - projects
  - ai
tags:
  - hackathon
  - google-cloud
  - multimodal-ai
  - llms
  - computer-vision
  - gemma
  - healthcare
classes: wide
toc: true
toc_label: "On this page"
toc_icon: "brain"
---

## Overview

Last month, I participated in the **Google Cloud AI Hackathon**, a fast-paced competition focused on building applied AI systems with real deployment constraints.

Our project, **MedAnnotator**, was selected as one of the winning teams.  
The official announcement is available here:  
👉 https://opendatascience.com/highlighting-the-winners-of-the-december-2025-google-cloud-ai-hackathon/

The problem we addressed is well known in medical AI:

> Medical image annotation is expensive, slow, inconsistent, and difficult to scale, yet it remains a critical dependency for clinical workflows, research, and model development.

Our goal was to design a system that prioritizes **correctness, structure, and deployability**, rather than novelty.

---

## Problem Definition

We identified three core bottlenecks in existing medical image annotation workflows:

1. **Limited scalability**: Annotation relies heavily on expert time, which does not scale with data volume.

2. **High label variance**: Inter-annotator disagreement introduces noise and reduces downstream model reliability.

3. **Unstructured outputs**: Free-text annotations are difficult to validate, audit, or integrate into pipelines.

The system was designed to generate **structured, reviewable annotations** from medical images while explicitly supporting human oversight.

---

## System Architecture

### 1. Two-Tier Model Design

Rather than relying on a single model, we decomposed the task:

- **MedGemma**: Responsible for domain-specific medical image understanding. This model handled image-level reasoning and feature extraction.

- **Gemini (API)**: Used for validation, reasoning over MedGemma outputs, and producing structured, schema-compliant annotations.

This separation reduced coupling, improved iteration speed, and made failure modes easier to reason about.

---

### 2. Structured Outputs as a First-Class Constraint

All annotations were generated using predefined schemas.  
We deliberately avoided free-form text.

This enabled:
- Deterministic validation
- Easier human review and correction
- Immediate downstream usability (storage, analytics, retraining)

Structured outputs also simplified debugging during the demo and made model behavior more transparent.

---

### 3. Human-in-the-Loop by Design

The workflow explicitly supported:
1. Model-generated initial annotations
2. Human review and edits
3. Auditable final outputs

In a healthcare context, this tradeoff is intentional.  
Reviewability and traceability matter more than full autonomy.

---

### 4. Deployment-Oriented Decisions

MedGemma is computationally heavy, so we deployed it on cloud compute to:
- Keep latency within interactive bounds
- Avoid blocking UI workflows
- Enable rapid iteration during the hackathon

This allowed us to focus on system behavior and evaluation rather than infrastructure limitations.

---

## Why the Project Worked

Several factors contributed to the outcome:

- **Tight scope**: We focused on a concrete bottleneck instead of building a generic platform.

- **Clear system boundaries**: Each component had a well-defined responsibility.

- **Production mindset**: Latency, structure, and deployment were treated as core requirements, not afterthoughts.

- **Constraint-driven design**: Hackathon limits forced architectural clarity.

---

## Key Takeaways

This experience reinforced several principles that consistently hold in applied AI:

- System design often matters more than model size.
- Structured outputs outperform clever prompts in production settings.
- Human-in-the-loop workflows remain essential in high-stakes domains.
- Deployment constraints improve, rather than limit, design quality.

---

## Future Extensions

The prototype can be extended in several directions:
- Batch annotation pipelines for large-scale datasets
- Integration with PACS or clinical data systems
- Active learning loops using corrected annotations
- Quantitative evaluation tooling for annotation quality and drift

If you are working on applied multimodal systems, especially in healthcare, I am always open to technical discussions.

---

*Thanks to my teammates, ODSC, and Google Cloud for running a technically rigorous and well-executed hackathon.*
