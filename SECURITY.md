# Security policy

fhir-healthcare-ai is a research and engineering demonstration. It runs on synthetic
data and is not intended to be deployed against real patient records. Security reports
are still taken seriously, because the project's point is to show how a language model
can be kept inside a safe boundary over clinical data.

## Supported versions

Only the latest commit on `main` receives fixes.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/martinoa2000/fhir-healthcare-ai/security/advisories/new)
("Security" tab, "Report a vulnerability"). Do not open a public issue or pull request.

Include what you can of:

- the affected component (validator, query builder, FHIR client, API, web UI, ...),
- the steps or the question text that reproduce it, against the in-memory demo
  (`make demo`) where possible,
- what you expected and what happened.

You should receive an acknowledgement within 7 days. Once a fix is available, the
advisory is published with credit to the reporter unless you prefer otherwise.

## What counts

In scope, for example:

- a question or plan that makes the pipeline issue a request other than a GET of an
  allowlisted resource and parameter, or reach a different origin;
- a way around API-key authentication or the rate limiter;
- a key, question text or other secret written to logs or the audit trail when it
  should not be;
- script injection in the web UI, or formula injection in the CSV export.

Out of scope: findings that need `FHIR_ALLOW_WRITE=true` or a deliberately insecure
configuration, denial of service by volume, and the clinical accuracy of outputs (the
project makes no clinical claim; open a regular issue for correctness bugs).

Never include real patient data in a report.
