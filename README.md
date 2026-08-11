<p align="center">
  <h1 align="center">Does Chain-of-Thought Prefer a Different Persona</h1>
  <p align="center"><strong>Compare persona markers in final answers versus intermediate reasoning traces.</strong></p>
</p>

---

## Overview

This repository implements experimental profiles for **Does Chain-of-Thought Prefer a Different Persona**. Config, caching, hooks, metrics, ablations, reporting, and CI are built for reproducible local pilots on small open-weight models.

Hypothesis (one line): Compare persona markers in final answers versus intermediate reasoning traces.

## Motivation

Interpretability and safety claims fail in practice for boring engineering
reasons: unpinned weights, chat templates skipped, invalid layer indices,
intervals that span zero treated as nulls, and stages that raise
`NotImplementedError`. This repo treats those as first-class bugs.

## Status

Focus: compare persona markers in final answers versus intermediate reasoning traces. Shared infrastructure is in place; domain stages
must pass harness validation before any measured claim.

| Command | Purpose |
|---|---|
| `make install-dev` | editable install + pinned requirements |
| `make test` | full unit suite |
| `make ci` | lint + test + typecheck + api-contract + coverage |
| `make pilot` | end-to-end pilot profile |
| `make doctor` | environment / device report |
