# AI Co-Builder README

> **This document is superseded by the root [README.md](../README.md).**

This file served as the original build specification during initial development (v0.1-v0.2). The architecture has since evolved significantly:

- **Debate → Contribution Synthesis** (models never see each other's outputs)
- **92-142 LLM calls → 12-22 calls** per run
- **GPT removed** → replaced with Mistral Large
- **Heuristic filters → Model-evaluated Lab Gate** (Perplexity sonar-pro)

See the root README.md for current architecture, usage, and results.
