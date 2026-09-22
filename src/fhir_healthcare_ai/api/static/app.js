// fhir-healthcare-ai browser UI.
//
// Plain DOM code, no framework and no build step. Every value that comes from the API is
// inserted with textContent or as an attribute through el(); nothing from a response is
// ever parsed as HTML. The Content-Security-Policy set by the server backs this up.
"use strict";

const FALLBACK_DISCLAIMER =
  "Research and engineering demonstration only. Outputs are not validated for clinical " +
  "use and must not be used to make or support decisions about the care of any person.";

// ---------------------------------------------------------------------------- helpers

/** Create an element. `props` sets properties (className, type, ...); children may be
 *  nodes, strings (inserted as text) or null/undefined (skipped). */
function el(tag, props, ...children) {
  const node = document.createElement(tag);
  if (props) {
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null) continue;
      if (key === "dataset") Object.assign(node.dataset, value);
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (key in node) node[key] = value;
      else node.setAttribute(key, String(value));
    }
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function clear(node) {
  node.replaceChildren();
  return node;
}

function fmtDate(value) {
  if (!value) return "";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? String(value) : d.toISOString().slice(0, 10);
}

function fmtNumber(value, digits = 1) {
  if (value === null || value === undefined || value === "") return "";
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits).replace(/\.0+$/, "") : String(value);
}

function fmtValue(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : fmtNumber(value, 2);
  if (Array.isArray(value)) return value.length ? value.map(fmtValue).join(", ") : "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function table(headers, rows) {
  return el(
    "div",
    { className: "table-wrap" },
    el(
      "table",
      null,
      el("thead", null, el("tr", null, headers.map((h) => el("th", { scope: "col" }, h)))),
      el("tbody", null, rows),
    ),
  );
}

function callout(kind, title, items) {
  const box = el("div", { className: `callout ${kind}`, role: kind === "error" ? "alert" : null });
  if (title) box.append(el("strong", null, title));
  if (items && items.length) box.append(el("ul", null, items.map((i) => el("li", null, i))));
  return box;
}

function disclaimer(text) {
  return el("p", { className: "fine" }, text || FALLBACK_DISCLAIMER);
}

/** Turn a FastAPI error body into readable lines. `detail` is a string for errors the
 *  API raises itself, and a list of {loc, msg} objects for request validation errors. */
function errorLines(body, status) {
  const detail = body && body.detail;
  if (typeof detail === "string") return [detail];
  if (Array.isArray(detail)) {
    return detail.map((d) => {
      const loc = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
      return loc ? `${loc}: ${d.msg}` : String(d.msg || JSON.stringify(d));
    });
  }
  return [`HTTP ${status}`];
}

class ApiError extends Error {
  constructor(status, lines, correlationId) {
    super(lines.join("; "));
    this.status = status;
    this.lines = lines;
    this.correlationId = correlationId;
  }
}

// The API key lives in sessionStorage: it survives a reload but not the tab, and is never
// written into the page. Storage can be unavailable (private mode), so every access is
// guarded and the key then simply lasts until reload.
const KEY_STORAGE = "fhir-ai-api-key";
let apiKey = "";
try {
  apiKey = sessionStorage.getItem(KEY_STORAGE) || "";
} catch (_) {
  apiKey = "";
}

function setApiKey(value) {
  apiKey = value.trim();
  try {
    if (apiKey) sessionStorage.setItem(KEY_STORAGE, apiKey);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch (_) {
    /* storage unavailable: keep it in memory only */
  }
}

async function api(path, options = {}) {
  const headers = { accept: "application/json", ...(options.headers || {}) };
  if (apiKey) headers["X-API-Key"] = apiKey;
  let response;
  try {
    response = await fetch(path, { ...options, headers });
  } catch (err) {
    throw new ApiError(0, [`Network error: ${err.message}`], null);
  }
  let body = null;
  try {
    body = await response.json();
  } catch (_) {
    body = null;
  }
  if (!response.ok) {
    const id = (body && body.correlation_id) || response.headers.get("X-Request-ID");
    throw new ApiError(response.status, errorLines(body, response.status), id);
  }
  return body;
}

function renderError(target, err) {
  const title =
    err.status === 422 ? "The request could not be answered (422)"
    : err.status === 503 ? "A dependency is unavailable (503)"
    : err.status === 404 ? "Not found (404)"
    : err.status ? `Request failed (${err.status})`
    : "Request failed";
  const box = callout("error", title, err.lines || [String(err)]);
  if (err.correlationId) box.append(el("div", { className: "meta" }, `request id ${err.correlationId}`));
  clear(target).append(box, disclaimer());
}

async function withBusy(button, target, work) {
  button.disabled = true;
  clear(target).append(el("p", { className: "muted" }, "Working…"));
  try {
    await work();
  } catch (err) {
    renderError(target, err);
  } finally {
    button.disabled = false;
  }
}

// ----------------------------------------------------------------------------- health

async function loadHealth() {
  const box = document.getElementById("health");
  try {
    const h = await api("/health");
    const statusKind = h.status === "ok" ? "ok" : h.status === "degraded" ? "warn" : "bad";
    const badges = [
      el("span", { className: `badge ${statusKind}`, title: `version ${h.version}` }, h.status),
      el(
        "span",
        { className: `badge ${h.fhir.reachable ? "ok" : "bad"}`, title: h.fhir.base_url },
        `FHIR ${h.fhir.reachable ? "reachable" : "unreachable"}${h.fhir.in_memory ? " (in-memory)" : ""}`,
      ),
      el(
        "span",
        { className: "badge", title: `configured: ${h.llm.configured}` },
        `LLM ${h.llm.active} · ${h.llm.model}${h.llm.local ? " · local" : " · hosted"}`,
      ),
    ];
    if (h.llm.fallback) {
      badges.push(
        el(
          "span",
          { className: "badge warn", title: h.llm.reason || "" },
          `degraded: fallback from ${h.llm.configured} to ${h.llm.active}`,
        ),
      );
    }
    clear(box).append(...badges);
  } catch (err) {
    clear(box).append(el("span", { className: "badge bad", title: err.message }, "API unreachable"));
  }
}

// ------------------------------------------------------------------------------ query

async function loadExamples() {
  const box = document.getElementById("examples");
  try {
    const caps = await api("/capabilities");
    const question = document.getElementById("question");
    clear(box).append(
      ...(caps.example_questions || []).map((q) =>
        el(
          "button",
          {
            type: "button",
            className: "chip",
            onclick: () => {
              question.value = q;
              question.focus();
            },
          },
          q,
        ),
      ),
    );
  } catch (_) {
    clear(box);
  }
}

function evidenceTable(evidence) {
  if (!evidence || !evidence.length) return el("p", { className: "muted" }, "No evidence attached.");
  return table(
    ["Resource", "Display", "Value", "Date"],
    evidence.map((e) =>
      el(
        "tr",
        null,
        el("td", { className: "mono" }, `${e.resource_type}/${e.resource_id}`),
        el("td", null, e.display || e.concept || "", e.note ? el("div", { className: "meta" }, e.note) : null),
        el("td", null, e.value || ""),
        el("td", null, fmtDate(e.effective)),
      ),
    ),
  );
}

function patientsTable(patients) {
  const rows = [];
  for (const p of patients) {
    const detail = el(
      "tr",
      { className: "detail", hidden: true },
      el("td", { colSpan: 5 }, p.summary ? el("p", null, p.summary) : null, evidenceTable(p.evidence)),
    );
    const toggle = el(
      "button",
      {
        type: "button",
        className: "link",
        "aria-expanded": "false",
        title: "Show evidence",
        onclick: () => {
          detail.hidden = !detail.hidden;
          toggle.setAttribute("aria-expanded", String(!detail.hidden));
          toggle.textContent = detail.hidden ? "▸" : "▾";
        },
      },
      "▸",
    );
    const analyse = el(
      "button",
      { type: "button", className: "link", onclick: () => analysePatient(p.patient_id) },
      "analyse",
    );
    rows.push(
      el(
        "tr",
        null,
        el("td", null, toggle),
        el("td", { className: "mono" }, p.patient_id, " ", analyse),
        el("td", null, fmtNumber(p.age_years, 0)),
        el("td", null, p.gender || ""),
        el("td", null, (p.matched_steps || []).join(", ")),
      ),
      detail,
    );
  }
  return table(["", "Patient", "Age", "Gender", "Matched steps"], rows);
}

function renderQuery(target, r) {
  const plan = r.query_plan || {};
  const out = [];

  if (plan.unsupported) {
    out.push(
      callout("warn", "This question is outside what the system can answer safely.", [
        plan.unsupported_reason || "No reason was given.",
      ]),
    );
  }

  if (r.narrative) out.push(el("h3", null, "Answer"), el("p", { className: "narrative" }, r.narrative));

  const issues = (r.validation_issues || []).map(
    (i) => `${i.severity}: ${i.message}${i.location ? ` (${i.location})` : ""}`,
  );
  if (r.warnings && r.warnings.length) out.push(callout("warn", "Warnings", r.warnings));
  if (issues.length) out.push(callout("warn", "Validation", issues));
  if (plan.assumptions && plan.assumptions.length) {
    out.push(callout("info", "Assumptions", plan.assumptions));
  }

  const cohort = r.cohort || {};
  const trace = r.trace || {};
  const meta = [
    `${cohort.total_patients ?? (r.patients || []).length} patients`,
    `${cohort.total_resources ?? 0} resources`,
    plan.intent ? `intent ${plan.intent}` : null,
    r.analysis_type && r.analysis_type !== "none" ? `analysis ${r.analysis_type}` : null,
    trace.fhir_requests !== undefined ? `${trace.fhir_requests} FHIR requests` : null,
    trace.llm_calls !== undefined ? `${trace.llm_calls} LLM calls` : null,
    trace.truncated ? "truncated" : null,
  ].filter(Boolean);
  if (!plan.unsupported) out.push(el("p", { className: "meta" }, meta.join(" · ")));

  if (r.fhir_queries && r.fhir_queries.length) {
    out.push(
      el("h3", null, "FHIR queries executed"),
      el("ul", { className: "queries" }, r.fhir_queries.map((q) => el("li", { className: "mono" }, q))),
    );
  }

  if (r.patients && r.patients.length) {
    out.push(el("h3", null, "Patients"), patientsTable(r.patients));
  } else if (!plan.unsupported) {
    out.push(el("p", { className: "muted" }, "No patients matched."));
  }

  // A single-patient question (a record summary) gets its analysis inline; for a cohort
  // the per-patient analysis is one click away ("analyse") instead of a wall of tables.
  if (r.analyses && r.analyses.length === 1) {
    const a = r.analyses[0];
    out.push(el("h3", null, `Analysis · ${a.patient_id}`), analysisBody(a));
  }

  if (trace.correlation_id) out.push(el("p", { className: "meta" }, `request id ${trace.correlation_id}`));
  out.push(disclaimer(r.disclaimer));
  clear(target).append(...out);
}

function setupQueryForm() {
  const form = document.getElementById("query-form");
  const target = document.getElementById("query-result");
  const button = document.getElementById("query-submit");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const body = {
      question: document.getElementById("question").value.trim(),
      narrate: document.getElementById("narrate").checked,
    };
    const asOf = document.getElementById("as-of").value;
    if (asOf) body.as_of = asOf;
    withBusy(button, target, async () => {
      const r = await api("/query", {
        method: "POST",
        headers: { "content-type": "application/json", accept: "application/json" },
        body: JSON.stringify(body),
      });
      renderQuery(target, r);
    });
  });
}

// ---------------------------------------------------------------------------- patient

function analysisBody(a) {
  const parts = [];
  parts.push(
    el(
      "dl",
      { className: "kv" },
      el("dt", null, "Age"),
      el("dd", null, fmtNumber(a.age_years, 0) || "—"),
      el("dt", null, "Gender"),
      el("dd", null, a.gender || "—"),
    ),
  );

  const risk = a.risk;
  if (risk) {
    const kind = risk.band === "high" ? "bad" : risk.band === "moderate" ? "warn" : "ok";
    parts.push(
      el("h3", null, "Risk"),
      el(
        "p",
        null,
        el("span", { className: `badge ${kind}` }, `${risk.band}`),
        ` score ${fmtNumber(risk.score, 2)} · ${risk.model_name} ${risk.model_version}`,
      ),
    );
    if (risk.contributing_factors && risk.contributing_factors.length) {
      parts.push(callout("info", "Contributing factors", risk.contributing_factors));
    }
    if (risk.missing_features && risk.missing_features.length) {
      parts.push(el("p", { className: "meta" }, `Missing inputs: ${risk.missing_features.join(", ")}`));
    }
  }

  parts.push(el("h3", null, "Abnormal labs"));
  if (a.abnormal_labs && a.abnormal_labs.length) {
    parts.push(
      table(
        ["Lab", "Value", "Flag", "Reference", "Date", "Resource"],
        a.abnormal_labs.map((l) => {
          const ref = [l.reference_low, l.reference_high].map((v) => fmtNumber(v, 2));
          return el(
            "tr",
            null,
            el("td", null, l.display || l.concept),
            el("td", null, `${fmtNumber(l.value, 2)} ${l.unit || ""}`.trim()),
            el("td", null, l.flag),
            el("td", null, ref[0] || ref[1] ? `${ref[0] || "…"} – ${ref[1] || "…"}` : ""),
            el("td", null, fmtDate(l.effective)),
            el("td", { className: "mono" }, `${l.evidence.resource_type}/${l.evidence.resource_id}`),
          );
        }),
      ),
    );
  } else {
    parts.push(el("p", { className: "muted" }, "None found."));
  }

  if (a.data_gaps && a.data_gaps.length) parts.push(callout("warn", "Data gaps", a.data_gaps));

  const features = Object.entries(a.features || {});
  if (features.length) {
    // The feature contract has ~100 entries; fold it so risk and labs stay in view.
    parts.push(
      el(
        "details",
        { className: "features" },
        el("summary", null, `Features (${features.length})`),
        table(
          ["Feature", "Value"],
          features.map(([k, v]) =>
            el("tr", null, el("td", { className: "mono" }, k), el("td", null, fmtValue(v))),
          ),
        ),
      ),
    );
  }
  return el("div", null, parts);
}

async function analysePatient(patientId) {
  const input = document.getElementById("patient-id");
  const asOf = document.getElementById("patient-as-of");
  const queryAsOf = document.getElementById("as-of").value;
  input.value = patientId;
  if (queryAsOf && !asOf.value) asOf.value = queryAsOf;
  document.getElementById("patient-form").requestSubmit();
  document.getElementById("patient-title").scrollIntoView({ behavior: "smooth", block: "start" });
}

function setupPatientForm() {
  const form = document.getElementById("patient-form");
  const target = document.getElementById("patient-result");
  const button = document.getElementById("patient-submit");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const id = document.getElementById("patient-id").value.trim();
    const asOf = document.getElementById("patient-as-of").value;
    const query = asOf ? `?as_of=${encodeURIComponent(asOf)}` : "";
    withBusy(button, target, async () => {
      const a = await api(`/patient/${encodeURIComponent(id)}/analyze${query}`);
      clear(target).append(
        el("h3", null, `Patient ${a.patient_id}`),
        analysisBody(a),
        disclaimer(a.risk && a.risk.disclaimer),
      );
    });
  });
}

// ------------------------------------------------------------------------------- boot

function setupKeyForm() {
  const form = document.getElementById("key-form");
  const input = document.getElementById("api-key");
  input.value = apiKey;
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    setApiKey(input.value);
    loadHealth();
    loadExamples();
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupKeyForm();
  setupQueryForm();
  setupPatientForm();
  loadHealth();
  loadExamples();
});
