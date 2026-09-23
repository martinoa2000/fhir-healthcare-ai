# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- `LICENSE` (Apache-2.0), `THIRD_PARTY_NOTICES.md`, `SECURITY.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, issue and pull request templates, and Dependabot updates for
  pip, GitHub Actions and Docker.

### Fixed

- The deterministic planner's keyword fallbacks (reduced kidney function, abnormal
  potassium, elevated HbA1c) no longer answer a question that also names an age, a sex,
  a drug, an encounter or a second diagnosis. "Which diabetic patients 65 or older have
  diabetic nephropathy?" is now refused instead of returning the whole
  reduced-kidney-function cohort.

## [0.1.0]

First public version.

### Added

- Governed question-to-cohort pipeline over HL7 FHIR R4: LLM planner emitting a
  structured `QueryPlan`, concept expander, allowlist validator, deterministic query
  builder and read-only FHIR client.
- LLM providers: self-hosted vLLM (default), Hugging Face, OpenAI, Anthropic, and a
  deterministic rule-based planner with automatic, reported fallback.
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
  truth, refusals and safety violations; gated in CI.
- Docker image and Compose stack with HAPI FHIR, seeder and optional vLLM.

[Unreleased]: https://github.com/martinoa2000/fhir-healthcare-ai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/martinoa2000/fhir-healthcare-ai/releases/tag/v0.1.0
