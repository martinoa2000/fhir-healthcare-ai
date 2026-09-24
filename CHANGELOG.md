# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Changed

- Web UI: the derivation is drawn as a cohort flow diagram, one box per search with
  its patient count, exclusions branching off the main line and the resulting cohort as
  the last box. The question, the patient count and the query cost head the results;
  the question box holds its options and the run button; the header shows the FHIR
  server and the planner as compact status labels.

## [0.1.0] - 2026-09-23

First public version.

### Added

- Governed question-to-cohort pipeline over HL7 FHIR R4: LLM planner emitting a
  structured `QueryPlan`, concept expander, allowlist validator, deterministic query
  builder and read-only FHIR client.
- Local-only inference: a self-hosted vLLM server serving Qwen3.5-9B with thinking mode
  off. On a 16 GB Apple Silicon Mac, vllm-metal serves the 4-bit MLX weights
  (`mlx-community/Qwen3.5-9B-MLX-4bit`, `make serve-model`); the compose `vllm`
  profile serves the original weights on a 24 GB NVIDIA GPU. Also included is a
  deterministic rule-based planner used by the tests and as an automatic, reported fallback.
  No hosted-API provider exists, so question text never leaves the deployment.
- The rule-based planner refuses any question with a criterion it cannot express,
  rather than answering part of it.
- Normalisation, per-patient features, abnormal-lab detection and a transparent risk
  score; evidence links from every claim to its FHIR resource.
- FastAPI service: `/query`, `/query/export` (CSV, FHIR `Group`, FHIR `Bundle`),
  `/patient/{id}/analyze`, `/capabilities`, health probes and `/audit`.
- API-key authentication, per-caller rate limiting and a JSONL audit trail with
  correlation ids.
- Dependency-free web UI served at `/`.
- Seeded synthetic population generator and loader (`fhir-ai-seed`), in-memory FHIR
  server for tests and demos.
- `fhir-ai-bench` benchmark scoring cohorts against independently computed ground
  truth, refusals and safety violations; exits non-zero on failure.
- Docker image and Compose stack with HAPI FHIR, seeder and optional vLLM.
- Apache-2.0 `LICENSE`, third-party notices, security policy, contributing guide and
  code of conduct.

[Unreleased]: https://github.com/martinoa2000/fhir-healthcare-ai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/martinoa2000/fhir-healthcare-ai/releases/tag/v0.1.0
