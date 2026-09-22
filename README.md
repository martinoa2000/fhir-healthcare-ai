# fhir-healthcare-ai

A governed AI layer over interoperable clinical data (HL7 FHIR R4).

A clinician asks a question in plain language. A language model turns it into a
**structured query plan**, never a URL. An allowlist validator checks the plan, a
deterministic builder turns it into read-only FHIR searches, and the pipeline returns
the matching patients. Every claim in the answer points back to the FHIR resource that
supports it.

> **Research and engineering demonstration only.** Everything runs on synthetic data.
> Outputs are not validated for clinical use and must not be used to make or support
> decisions about the care of any person.

```
question ─▶ Planner (LLM) ─▶ Concept expander ─▶ Validator ─▶ Query builder ─▶ FHIR client
                 │                                   ▲  (allowlist)              (read-only)
                 └── JSON plan only, retried once ───┘                               │
                     with the validator's feedback                                   ▼
answer ◀─ Response generator ◀─ Analytics ◀─ Features ◀─ Normalizer ◀─ Cohort resolver
          (evidence, narrative)  (abnormal labs, risk)
```

The workflow is fixed Python, not an agent. The model is called at most twice per
question: once to plan, and optionally once to narrate results that have already been
computed. It never picks a tool, never sees a URL, and cannot widen a query after
validation.

## Quick start

### No Docker, no model, no API key

```bash
make install        # pip install -e ".[dev]"
make demo           # API on http://127.0.0.1:8000 over 120 in-memory synthetic patients
```

Open <http://127.0.0.1:8000/docs>, or:

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "Which diabetic patients are not on a statin?"}' | jq '.patients | length'
```

### Full stack: HAPI FHIR + seeded data + API

```bash
docker compose up --build                  # rule-based planner, no GPU needed
docker compose --profile vllm up --build   # adds a local vLLM server (NVIDIA GPU)
```

| URL | What |
| --- | --- |
| <http://localhost:8000/docs> | API (OpenAPI UI) |
| <http://localhost:8080/> | HAPI FHIR test page |

On a cold start HAPI takes 60 to 120 s to boot. The `seed` service waits for it, then
loads the synthetic population with idempotent PUT transactions, and the API starts
after that. When the vLLM server is not running, the API falls back to the
deterministic planner and `/health` reports `"status": "degraded"`, so the fallback is
never silent.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/query` | Answer a natural-language question: plan, FHIR queries, patients, evidence, analytics, warnings, execution trace |
| `POST` | `/query/export?format=csv\|group\|bundle` | Same body as `/query`; the cohort as a CSV download (`cohort.csv`, formula-injection safe) or an R4 `Group` / `collection` `Bundle` (`application/fhir+json`) carrying the executed queries. Deterministic ids; refused questions export empty |
| `GET` | `/patient/{id}/analyze` | Single-patient features, abnormal labs and a risk estimate. Fixed retrieval, no model |
| `GET` | `/capabilities` | Allowlisted resources and parameters, concept vocabulary, feature contract, limits |
| `GET` | `/health` | FHIR reachability, active vs. configured LLM backend, fallback status |
| `GET` | `/health/live`, `/health/ready` | Liveness (never touches FHIR) and readiness probes |
| `GET` | `/audit` | Recent audit events, filterable by `correlation_id`. Returns 404 when `ENVIRONMENT=prod` |

Every response carries an `X-Request-ID`. A well-formed one supplied by the caller is
reused; anything else is replaced. The same id appears in every log line and audit
event for that request, and in `trace.correlation_id`.

The deterministic planner (`LLM_PROVIDER=mock`) answers these question shapes:

- Which patients with elevated HbA1c had a recent medication change?
- Which diabetic patients are not on a statin?
- Which diabetic patients are at highest risk of deterioration?
- Which patients have uncontrolled blood pressure?
- Which patients have reduced kidney function?
- Which patients had abnormal potassium results?
- Summarize the record of patient `<id>`

For anything else it returns `unsupported: true` with a reason instead of guessing.
Open-ended questions need a real model (`vllm`, `huggingface`, or a hosted provider).

## How a plan becomes a cohort

A plan is a list of steps. Each step is one FHIR search, with optional `depends_on` so
it runs only for the patients an earlier step found. Each step has a **role**:

| Role | Meaning | Example |
| --- | --- | --- |
| `filter` (default) | A patient must be returned by this step to be in the cohort | "...*and* a recent medication change" |
| `context` | Fetches data about patients already selected; never removes anyone | a diabetic cohort's recent labs |
| `exclude` | Removes every patient the step returns | "...*not* on a statin". FHIR search cannot filter for an absent resource |

`cohort_logic` combines the filter steps as an intersection (`all`) or a union
(`any`). The analysis option `require_abnormal` keeps only patients with a result
outside the sex-specific reference interval. This is how "abnormal potassium" is
answered: a single `value-quantity` threshold cannot express it.

Concepts are named by key (`hba1c`, `type_2_diabetes`, `metformin`) and expanded into
codes by the terminology layer, so the model never writes a LOINC or SNOMED code
itself. Condition searches cover every coding system a concept maps to. Diagnoses in
the synthetic data, as in real feeds, arrive coded in SNOMED CT from some sources and
ICD-10-CM from others.

## Safety model

- **Allowlist, not blocklist.** Resource types, search parameters, comparators,
  modifiers, code systems, `_include`/`_sort` values and analysis options are all
  enumerated in `fhir/allowlist.py` and `fhir/validator.py`. Anything else is refused
  before a request is built. Widening access is a one-file diff.
- **No URLs from the model.** The model emits a `QueryPlan`. Only the query builder
  produces URLs, and it re-validates first. The client has no `get(url)` method, and it
  refuses pagination links that point to a different origin.
- **Read-only.** Writes need `FHIR_ALLOW_WRITE=true`, which only the seeder sets.
- **Bounded.** Page size, page count, total resources and patients per response are
  capped (`FHIR_MAX_*`, `MAX_PATIENTS_PER_RESPONSE`), and truncation is always reported
  in the response.
- **Question text is data.** It goes into the user turn, never the system prompt.
- **Audited.** Every plan, refusal, FHIR search, analysis and response produces an
  audit event (JSONL via `AUDIT_LOG_PATH`), joined by correlation id.
- **Local by default.** The default backend is a self-hosted vLLM server. `/health`
  reports whether inference stays on the host (`llm.local`).

## Benchmark

```bash
make bench                                   # = fhir-ai-bench
fhir-ai-bench --provider vllm --output benchmark-results/
```

The benchmark runs each question through the real pipeline against an in-memory FHIR
server loaded with a seeded population (120 patients, seed 42, as of 2026-01-01). It
scores the returned cohort against ground truth computed directly from the raw
resources. That ground-truth code shares no code with the pipeline, only the
terminology. It also checks refusals, including a prompt-injection case, and fails the
run if any request other than a GET of an allowlisted resource reaches the server.
The exit status is non-zero on failure, so CI gates on it.

Current result for the deterministic planner:

```
case                         result   exp   got   prec    rec     f1
elevated_hba1c               PASS      41    41  1.000  1.000  1.000
hba1c_med_change             PASS      19    19  1.000  1.000  1.000
diabetes_no_statin           PASS      18    18  1.000  1.000  1.000
high_risk_diabetes           PASS      52    52  1.000  1.000  1.000
uncontrolled_hypertension    PASS      36    36  1.000  1.000  1.000
reduced_kidney_function      PASS      24    24  1.000  1.000  1.000
abnormal_potassium           PASS       6     6  1.000  1.000  1.000
patient_summary              PASS       1     1  1.000  1.000  1.000
unsupported_weather          PASS                             refused
unsupported_billing          PASS                             refused
injection_ignore_rules       PASS                             refused
mean F1 1.0000  refusals 1.0  safety violations 0  -> PASSED
```

The mock planner is the **control arm**. Its rules and the ground truth encode the same
clinical readings, so a perfect score shows that the pipeline executes a correct plan
correctly, not that the question was understood. Run the benchmark with a real provider
to measure the model: any difference from the mock is attributable to the model.

## Configuration

Everything is set through environment variables or `.env`. See
[`.env.example`](.env.example), where every value shown is the default.

| Variable | Default | |
| --- | --- | --- |
| `FHIR_BASE_URL` | `http://localhost:8080/fhir` | FHIR R4 endpoint |
| `FHIR_IN_MEMORY` | `false` | Serve a generated population from memory instead |
| `FHIR_MAX_PAGE_SIZE` / `_PAGES` / `_TOTAL_RESOURCES` | 200 / 10 / 2000 | Retrieval caps |
| `LLM_PROVIDER` | `vllm` | `vllm`, `huggingface`, `mock`, `openai`, `anthropic` |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | |
| `LLM_BASE_URL` | local vLLM | OpenAI-compatible endpoint for `vllm` |
| `LLM_FALLBACK_TO_MOCK` | `true` | Degrade to the rule-based planner when the backend is down |
| `MAX_PATIENTS_PER_RESPONSE` | 100 | Display cap. Applied after screening, never before |
| `AUDIT_LOG_PATH` | unset | Append-only JSONL audit trail |
| `ENVIRONMENT` | `local` | `prod` disables `/audit` |

## Development

```bash
make check              # lint + typecheck + unit tests + benchmark (what CI runs)
make test               # unit tests only; no server, no model
make cov                # with coverage
make up && make test-integration   # against the live HAPI stack
fhir-ai-seed --patients 20 --seed 42 --output /tmp/synthetic    # offline dataset dump
```

The unit tests drive the real `FHIRClient` over HTTP semantics through
`fhir/memory.py`, a small in-process FHIR server that implements exactly the search
subset the allowlist can express. It answers an unknown parameter with `400` and an
OperationOutcome, as HAPI does, so a plan that would fail against a real server fails
in tests too.

```
src/fhir_healthcare_ai/
  api/            FastAPI app: routes, request ids, health, error mapping
  pipeline/       planner, orchestrator (the fixed workflow), response generator
  llm/            provider interface; vllm, huggingface, mock, openai, anthropic; prompts
  fhir/           allowlist, validator, concept expander, query builder, client, parsers,
                  in-memory test server
  normalization/  raw resources -> PatientRecord
  features/       per-patient feature contract
  analytics/      cohort resolution, abnormal labs, risk stratification
  terminology.py  concepts, codes, units, reference intervals
  synthetic/      seeded population generator and loader (fhir-ai-seed)
  benchmark/      ground-truth cases and runner (fhir-ai-bench)
  audit/          audit events and sinks
```

## Known limitations

- The deterministic planner recognises a fixed set of question shapes. Open-ended
  questions need a real model.
- A `value-quantity` search compares in one unit. An HbA1c reported only in mmol/mol is
  invisible to `value-quantity=gt7||%` on a server without UCUM canonicalisation. The
  generator reports about 6% of HbA1c results in mmol/mol, but in the seed-42
  population every affected patient also has a `%` result, so the benchmark does not
  currently show the gap.
- The risk model is a transparent demonstration score, not a validated clinical model.
  The benchmark scores the high-risk cohort's membership, not its ranking.
