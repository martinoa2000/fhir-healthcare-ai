# Contributing

Thanks for your interest. Bug reports, benchmark cases, planner rules and documentation
fixes are all welcome.

## Ground rules

- **No real patient data, ever.** Not in code, tests, fixtures, issues, pull requests or
  screenshots. Reproduce problems with the synthetic generator (`fhir-ai-seed`) or the
  in-memory demo (`make demo`).
- **Keep the safety model intact.** The model emits a plan, never a URL; everything the
  pipeline may request is enumerated in `fhir/allowlist.py` and `fhir/validator.py`.
  Widening the allowlist is a deliberate, reviewed change with a test for what it allows
  and what it still refuses.
- **Refuse rather than guess.** A planner rule that cannot express every criterion in a
  question must decline, not answer part of it.
- By participating you agree to follow the [code of conduct](CODE_OF_CONDUCT.md).

## Development setup

Python 3.11 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
make install        # pip install -e ".[dev]"
make check          # lint + typecheck + unit tests + benchmark
```

`make help` lists every target. `make demo` starts the API and web UI on an in-memory
synthetic population with no Docker, model or API key. `make up && make test-integration`
runs the integration tests against a live HAPI FHIR server.

## Pull requests

1. Open an issue first for anything larger than a small fix, so the approach can be
   agreed before you write it.
2. Branch from `main` and keep each pull request to one change.
3. Add or update tests. A new question shape needs a planner test, and usually a
   benchmark case in `benchmark/cases.py` whose ground truth is computed independently
   of the pipeline.
4. Run `make check` and `ruff format .` before pushing.
5. Update `README.md` and `CHANGELOG.md` when behaviour, configuration or the API
   changes.

## Style

- `ruff` for lint and formatting (line length 100), `mypy` with `disallow_untyped_defs`.
- Match the surrounding code: small functions, explicit types, docstrings that explain
  *why*, not *what*.
- Commit messages in the imperative mood ("Add ...", "Fix ..."), with a body when the
  reason is not obvious.

## Reporting security issues

See [SECURITY.md](SECURITY.md). Please do not open public issues for vulnerabilities.
