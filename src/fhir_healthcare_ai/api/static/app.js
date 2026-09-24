"use strict";

/*
 * Cohort Explorer: a thin client over the public API.
 *
 * Every value that came from the server reaches the page through textContent or an
 * attribute set by createElement -- never through HTML parsing -- so a patient id or a
 * narrative cannot inject markup. The server's CSP is the second line of defence.
 */

// ---------------------------------------------------------------------------- dom

const SVG_NS = "http://www.w3.org/2000/svg";

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  applyProps(node, props);
  appendAll(node, children);
  return node;
}

function svg(tag, attrs, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value !== undefined && value !== null) node.setAttribute(key, String(value));
  }
  appendAll(node, children);
  return node;
}

function applyProps(node, props) {
  for (const [key, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "className") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2), value);
    } else if (key in node && typeof value !== "string") node[key] = value;
    else node.setAttribute(key, value === true ? "" : String(value));
  }
}

function appendAll(node, children) {
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

// Like Element.append, but skips false/null/undefined so conditionals can be inlined.
function put(node, ...children) {
  appendAll(node, children);
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------- format

const integerFormat = new Intl.NumberFormat();

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (typeof value !== "number") return String(value);
  if (Number.isInteger(value)) return integerFormat.format(value);
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

function fmtDate(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function fmtAge(years) {
  return years === null || years === undefined ? "-" : `${Math.floor(years)}`;
}

function sexLabel(gender) {
  if (!gender) return "-";
  return gender.charAt(0).toUpperCase() + gender.slice(1);
}

function plural(count, one, many) {
  return `${integerFormat.format(count)} ${count === 1 ? one : many}`;
}

// ---------------------------------------------------------------------------- api

const KEY_STORAGE = "cohort-explorer-api-key";
const RECENT_STORAGE = "cohort-explorer-recent";

function readStorage(store, key, fallback) {
  try {
    const raw = store.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch (_) {
    return fallback;
  }
}

function writeStorage(store, key, value) {
  try {
    if (value === null) store.removeItem(key);
    else store.setItem(key, JSON.stringify(value));
  } catch (_) {
    /* private mode or storage disabled: keep the value in memory only */
  }
}

let apiKey = readStorage(sessionStorage, KEY_STORAGE, "") || "";

class ApiError extends Error {
  constructor(status, lines, requestId) {
    super(lines.join("; "));
    this.status = status;
    this.lines = lines;
    this.requestId = requestId;
  }
}

function errorLines(body, status) {
  const detail = body && body.detail;
  if (typeof detail === "string") return [detail];
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      const where = Array.isArray(item.loc) ? item.loc.filter((p) => p !== "body").join(".") : "";
      return where ? `${where}: ${item.msg}` : String(item.msg || item);
    });
  }
  if (status === 0) return ["The server could not be reached."];
  return [`The server answered ${status}.`];
}

async function request(path, options = {}) {
  const headers = { accept: "application/json", ...(options.headers || {}) };
  if (apiKey) headers["X-API-Key"] = apiKey;
  let response;
  try {
    response = await fetch(path, { ...options, headers });
  } catch (err) {
    throw new ApiError(0, [`Network error: ${err.message}`], null);
  }
  if (!response.ok) {
    let body = null;
    try {
      body = await response.json();
    } catch (_) {
      body = null;
    }
    const id = (body && body.correlation_id) || response.headers.get("X-Request-ID");
    if (response.status === 401) onUnauthorized();
    throw new ApiError(response.status, errorLines(body, response.status), id);
  }
  return response;
}

async function api(path, options) {
  const response = await request(path, options);
  return response.json();
}

// ---------------------------------------------------------------------------- state

const state = {
  lastBody: null,
  response: null,
  analyses: new Map(),
  sort: { key: "patient_id", dir: 1 },
  filter: "",
  queryRun: 0,
};

// ---------------------------------------------------------------------------- status

const WARNING_ICON = () =>
  svg("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" },
    svg("path", { d: "M12 3l10 18H2L12 3z" }),
    svg("path", { d: "M12 10v5M12 18h.01" }));

async function loadStatus() {
  const box = $("status");
  try {
    const health = await api("/health");
    const fhirOk = health.fhir.reachable;
    const planner = health.llm.active === "mock" ? "Rule-based" : health.llm.model;
    const items = [
      el("span", { className: "status-item", title: health.fhir.base_url },
        el("span", { className: `status-dot ${fhirOk ? "ok" : "bad"}`, "aria-hidden": "true" }),
        el("span", { className: "status-key", text: "FHIR R4" }),
        el("span", {
          text: !fhirOk ? "Unreachable" : health.fhir.in_memory ? "Demo data, in memory" : "Connected",
        })),
      el("span", { className: "status-item", title: `${health.llm.model} (configured: ${health.llm.configured})` },
        el("span", { className: "status-key", text: "Planner" }),
        el("span", { text: `${planner}, ${health.llm.local ? "local" : "hosted"}` })),
    ];
    if (health.llm.fallback) {
      items.push(el("span", { className: "status-flag", title: health.llm.reason || "" },
        WARNING_ICON(),
        `Using the rule-based planner: ${health.llm.configured} is unavailable`));
    }
    put(clear(box), ...items);
  } catch (err) {
    const locked = err.status === 401;
    put(clear(box), el("span", { className: "status-item" },
      el("span", { className: "status-dot bad", "aria-hidden": "true" }),
      locked ? "An API key is required" : "The API is unreachable"));
  }
}

function onUnauthorized() {
  const dialog = $("access-dialog");
  $("access-reason").textContent =
    "This server requires an API key. Enter the key you were given; it stays in this " +
    "browser tab and is sent as X-API-Key.";
  if (!dialog.open) dialog.showModal();
}

function setupAccess() {
  const dialog = $("access-dialog");
  const input = $("api-key");
  $("access-button").addEventListener("click", () => {
    input.value = apiKey;
    dialog.showModal();
    input.focus();
  });
  $("key-form").addEventListener("submit", () => {
    apiKey = input.value.trim();
    writeStorage(sessionStorage, KEY_STORAGE, apiKey || null);
    toast(apiKey ? "API key saved for this tab" : "API key cleared");
    loadStatus();
    loadLibrary();
  });
  $("key-clear").addEventListener("click", () => {
    apiKey = "";
    input.value = "";
    writeStorage(sessionStorage, KEY_STORAGE, null);
    dialog.close();
    toast("API key cleared");
    loadStatus();
  });
}

// ---------------------------------------------------------------------------- library

const TOPICS = [
  ["A single patient", /summari[sz]e|record of patient/i],
  ["Visits and admissions", /emergency|admitted|admission|visit/i],
  ["Kidney", /kidney|renal|egfr|ckd|nephro/i],
  ["Heart and blood pressure", /blood pressure|hypertens|antihypertens|heart|ldl|cholesterol/i],
  ["Diabetes", /diabet|hba1c|insulin|metformin|sglt2|glucose/i],
];

function topicOf(question) {
  for (const [name, pattern] of TOPICS) if (pattern.test(question)) return name;
  return "Labs and medications";
}

async function loadLibrary() {
  const box = $("library");
  let questions = [];
  try {
    questions = (await api("/capabilities")).example_questions || [];
  } catch (_) {
    clear(box);
    return;
  }
  const groups = new Map();
  for (const question of questions) {
    const topic = topicOf(question);
    if (!groups.has(topic)) groups.set(topic, []);
    groups.get(topic).push(question);
  }
  // Matching order puts the narrow single-patient rule first; display leads with the
  // topics that have the most questions.
  const order = ["Diabetes", "Heart and blood pressure", "Kidney", "Visits and admissions",
    "Labs and medications", "A single patient"];
  put(clear(box), 
    ...order.filter((name) => groups.has(name)).map((name) =>
      el("div", { className: "library-group" },
        el("h3", { text: name }),
        el("ul", null, groups.get(name).map((q) =>
          el("li", null, el("button", {
            type: "button",
            className: "question-link",
            text: q,
            onclick: () => runFromLibrary(q),
          })))))));
}

function runFromLibrary(question) {
  $("question").value = question;
  $("query-form").requestSubmit();
}

function renderRecent() {
  const recent = readStorage(localStorage, RECENT_STORAGE, []);
  const box = $("recent");
  if (!Array.isArray(recent) || recent.length === 0) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  put(clear(box), 
    el("h3", { text: "Recent questions" }),
    el("ul", null, recent.map((q) =>
      el("li", null, el("button", {
        type: "button", className: "question-link", text: q, onclick: () => runFromLibrary(q),
      })))));
}

function remember(question) {
  const recent = readStorage(localStorage, RECENT_STORAGE, []);
  const list = [question, ...(Array.isArray(recent) ? recent : []).filter((q) => q !== question)];
  writeStorage(localStorage, RECENT_STORAGE, list.slice(0, 6));
  renderRecent();
}

// ---------------------------------------------------------------------------- query

function setupQueryForm() {
  const form = $("query-form");
  const question = $("question");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = question.value.trim();
    if (text.length < 3) {
      question.focus();
      return;
    }
    const body = { question: text, narrate: $("narrate").checked };
    const asOf = $("as-of").value;
    if (asOf) body.as_of = asOf;
    runQuery(body);
  });
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
}

async function runQuery(body) {
  const run = ++state.queryRun;
  state.lastBody = body;
  state.filter = "";
  $("filter").value = "";
  writeUrl(body);
  remember(body.question);

  const button = $("run");
  button.disabled = true;
  button.textContent = "Running";
  document.querySelector(".ask").classList.add("compact");
  $("workspace").setAttribute("aria-busy", "true");
  announce("Running the query");
  // Most answers arrive in well under a quarter of a second; a skeleton that flashes
  // for 20ms reads as a glitch. It only appears when the wait is noticeable.
  const loadingTimer = setTimeout(() => {
    if (run === state.queryRun) showLoading(body.question);
  }, LOADING_DELAY_MS);

  try {
    const response = await api("/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (run !== state.queryRun) return;
    state.response = response;
    state.analyses = new Map((response.analyses || []).map((a) => [a.patient_id, a]));
    renderResponse(response);
  } catch (err) {
    if (run !== state.queryRun) return;
    showError(err);
  } finally {
    clearTimeout(loadingTimer);
    if (run === state.queryRun) {
      button.disabled = false;
      button.textContent = "Run query";
      $("workspace").setAttribute("aria-busy", "false");
    }
  }
}

const LOADING_DELAY_MS = 250;

function announce(message) {
  // Cleared first so the same sentence twice in a row is still read out.
  const region = $("announce");
  region.textContent = "";
  setTimeout(() => {
    region.textContent = message;
  }, 50);
}

function writeUrl(body) {
  const params = new URLSearchParams({ q: body.question });
  const patient = new URLSearchParams(location.search).get("patient");
  if (patient) params.set("patient", patient);
  if (body.as_of) params.set("as_of", body.as_of);
  try {
    history.replaceState(null, "", `?${params}`);
  } catch (_) {
    /* sandboxed documents may refuse history changes */
  }
}

function showLoading(question) {
  const workspace = $("workspace");
  workspace.hidden = false;
  workspace.setAttribute("aria-busy", "true");
  $("toolbar").hidden = true;
  $("narrative").hidden = true;
  clear($("result-message"));
  for (const id of ["view-patients", "view-profile", "view-queries"]) clear($(id));
  put(clear($("ledger")), 
    el("div", { className: "loading", style: undefined },
      el("p", { className: "question-echo", text: question }),
      el("p", { className: "muted", text: "Planning the query, validating it and reading the FHIR server" }),
      el("div", { className: "loading-line", "aria-hidden": "true" })));
  put(clear($("derivation")), 
    el("div", { className: "loading", "aria-hidden": "true" },
      el("div", { className: "skeleton" }),
      el("div", { className: "skeleton short" }),
      el("div", { className: "skeleton" })));
}

function showError(err) {
  $("workspace").hidden = false;
  clear($("ledger"));
  put(clear($("derivation")), el("p", { className: "muted", text: "No plan was produced." }));
  const title =
    err.status === 503 ? "The query planner is unavailable"
      : err.status === 422 ? "This question could not be turned into a safe query"
        : err.status === 401 ? "An API key is required"
          : err.status === 429 ? "Too many requests"
            : "The query failed";
  announce(title);
  const guidance =
    err.status === 503 ? "Try again in a moment, or ask your administrator to check the model server."
      : err.status === 422 ? "Rephrase it, or start from one of the questions in the library."
        : err.status === 429 ? "Wait a minute before running another query."
          : null;
  put(clear($("result-message")), 
    el("div", { className: "callout error", role: "alert" },
      el("h3", { text: title }),
      el("ul", null, err.lines.map((line) => el("li", { text: line }))),
      guidance && el("p", { text: guidance }),
      err.requestId && el("p", { className: "request-id" }, "Request id ", el("code", { text: err.requestId }))));
}

// ---------------------------------------------------------------------------- response

function renderResponse(response) {
  $("workspace").hidden = false;
  renderDerivation(response);
  renderLedger(response);

  const narrative = $("narrative");
  narrative.hidden = !response.narrative;
  narrative.textContent = response.narrative || "";

  const message = clear($("result-message"));
  if (response.query_plan.unsupported) {
    put(message, 
      el("div", { className: "callout caution" },
        el("h3", { text: "This question is outside what the planner can answer" }),
        el("p", { text: response.query_plan.unsupported_reason || "No reason was given." }),
        el("p", { className: "muted", text: "Nothing was read from the FHIR server. Try one of the questions in the library." })));
    $("toolbar").hidden = true;
    document.querySelector(".ask").classList.remove("compact");
    announce("This question is outside what the planner can answer");
    return;
  }
  if ((response.validation_issues || []).length) {
    put(message, 
      el("div", { className: "callout caution" },
        el("h3", { text: "The validator raised notes about this plan" }),
        el("ul", null, response.validation_issues.map((issue) =>
          el("li", { text: `${issue.message}${issue.location ? ` (${issue.location})` : ""}` })))));
  }

  $("toolbar").hidden = false;
  const found = (response.patients || []).length;
  announce(`${plural(found, "patient", "patients")} found`);
  renderPatients();
  renderProfile(response);
  renderQueries(response);
  selectTab("tab-patients", false);
}

const ROLE_LABELS = {
  filter: "Selects patients",
  context: "Adds context, never removes anyone",
  exclude: "Removes these patients",
};

function describeParam(param) {
  // `system|code` values are shown by code only: the system is in the assumptions list.
  const values = (param.values || []).map((v) => (v.includes("|") ? v.slice(v.lastIndexOf("|") + 1) : v));
  const shown = values.slice(0, 3).join(", ");
  const more = values.length > 3 ? ` and ${values.length - 3} more` : "";
  const op = param.comparator ? `${param.comparator} ` : "";
  const unit = param.unit ? ` ${param.unit}` : "";
  return [
    el("span", { className: "param-name", text: param.modifier ? `${param.name}:${param.modifier}` : param.name }),
    ` ${op}`,
    el("code", { text: shown }),
    `${more}${unit}`,
  ];
}

function renderDerivation(response) {
  const plan = response.query_plan;
  const box = clear($("derivation"));
  if (plan.unsupported || !plan.steps.length) {
    put(box, el("p", { className: "muted", text: "No query was planned." }));
    return;
  }
  const counts = (response.cohort && response.cohort.per_step_counts) || {};
  const index = new Map(plan.steps.map((step, i) => [step.step_id, i + 1]));
  put(box, 
    el("ol", { className: "steps" }, plan.steps.map((step) => {
      const role = step.role || "filter";
      const count = counts[step.step_id];
      return el("li", { className: `step role-${role}` },
        el("span", { className: "step-marker", "aria-hidden": "true" }),
        el("p", { className: "step-role", text: ROLE_LABELS[role] || role }),
        el("p", { className: "step-title", text: `${step.resource_type} search` }),
        step.purpose && el("p", { className: "step-purpose", text: step.purpose }),
        step.params.length > 0 && el("ul", { className: "step-params" },
          step.params.map((param) => el("li", null, describeParam(param)))),
        step.depends_on && el("p", {
          className: "step-purpose",
          text: `Runs only for the patients found in step ${index.get(step.depends_on) || step.depends_on}`,
        }),
        count !== undefined && stepCount(role, count));
    }),
    // Screening happens after retrieval, so it is not a plan step -- but it is where
    // "120 patients with a potassium result" becomes "6 with an abnormal one", and a
    // derivation that skipped it would not add up.
    plan.analysis.options && plan.analysis.options.require_abnormal && el("li", { className: "step role-screen" },
      el("span", { className: "step-marker", "aria-hidden": "true" }),
      el("p", { className: "step-role", text: "Keeps abnormal results only" }),
      el("p", { className: "step-title", text: "Reference interval screen" }),
      el("p", {
        className: "step-purpose",
        text: `A patient stays only with a ${(plan.analysis.concepts || []).join(", ") || "screened"} result outside the sex-specific reference interval.`,
      }))),
    cohortBox(response.cohort ? response.cohort.total_patients : (response.patients || []).length),
    el("p", {
      className: "logic",
      text: plan.cohort_logic === "any"
        ? "A patient is included when any selecting step finds them."
        : "A patient is included when every selecting step finds them.",
    }));

  const notes = [...(plan.assumptions || [])];
  for (const warning of response.warnings || []) if (!notes.includes(warning)) notes.push(warning);
  if (notes.length) {
    put(box, el("details", { className: "assumptions", open: true },
      el("summary", { text: `Assumptions and notes (${notes.length})` }),
      el("ul", null, notes.map((note) => el("li", { text: note })))));
  }
}

// The count is the headline of each box in the flow, as n is in a CONSORT diagram.
function stepCount(role, count) {
  const label = role === "exclude" ? "removed" : role === "context" ? "with data"
    : count === 1 ? "patient" : "patients";
  return el("p", { className: "step-count" },
    el("strong", { text: integerFormat.format(count) }),
    el("span", { text: label }));
}

function cohortBox(total) {
  return el("div", { className: "outcome" },
    el("p", { className: "outcome-label", text: "Cohort" }),
    el("p", { className: "step-count" },
      el("strong", { text: integerFormat.format(total) }),
      el("span", { text: total === 1 ? "patient" : "patients" })));
}

function renderLedger(response) {
  const trace = response.trace || {};
  const count = (response.patients || []).length;
  const total = response.cohort ? response.cohort.total_patients : count;
  const items = [
    ["FHIR requests", fmt(trace.fhir_requests)],
    ["Resources read", fmt(trace.resources_fetched)],
    ["Model calls", fmt(trace.llm_calls)],
    ["Time", trace.stages ? `${fmt(Object.values(trace.stages).reduce((a, b) => a + b, 0) / 1000, 2)} s` : "-"],
  ];
  put(clear($("ledger")), 
    el("p", { className: "question-echo", text: response.question }),
    el("div", { className: "ledger-row" },
      el("p", { className: "ledger-headline" },
        integerFormat.format(count),
        el("small", { text: count === 1 ? "patient" : "patients" })),
      el("dl", null, items.map(([label, value]) =>
        el("div", null, el("dt", { text: label }), el("dd", { text: value }))))),
    trace.truncated && el("p", { className: "callout caution", style: undefined },
      `Results were capped: ${total} matched, ${count} are shown. Narrow the question for the full set.`));
}

// ---------------------------------------------------------------------------- patients

const COLUMNS = [
  { key: "patient_id", label: "Patient" },
  { key: "age", label: "Age", num: true },
  { key: "sex", label: "Sex" },
  { key: "matched", label: "Found by", narrow: true },
  { key: "risk", label: "Risk", needs: "risk" },
  { key: "flags", label: "Out of range", needs: "labs" },
  { key: "evidence", label: "Evidence", num: true, narrow: true },
];

function patientRows() {
  const response = state.response;
  return (response.patients || []).map((match) => {
    const analysis = state.analyses.get(match.patient_id);
    const labs = analysis ? analysis.abnormal_labs || [] : [];
    return {
      match,
      analysis,
      patient_id: match.patient_id,
      age: match.age_years,
      sex: match.gender || "",
      matched: (match.matched_steps || []).length,
      risk: analysis && analysis.risk ? analysis.risk.score : null,
      high: labs.filter((lab) => lab.flag.includes("high")).length,
      low: labs.filter((lab) => lab.flag.includes("low")).length,
      flags: labs.length,
      evidence: (match.evidence || []).length,
    };
  });
}

function renderPatients() {
  const view = clear($("view-patients"));
  const rows = patientRows();
  const hasRisk = rows.some((row) => row.risk !== null);
  const hasLabs = state.analyses.size > 0;
  // With one selecting step every row says the same thing; the rail already says it.
  const variedMatch = new Set(rows.map((row) => (row.match.matched_steps || []).join("|"))).size > 1;
  const columns = COLUMNS.filter((c) =>
    c.needs === "risk" ? hasRisk
      : c.needs === "labs" ? hasLabs
        : c.key === "matched" ? variedMatch
          : true);

  const needle = state.filter.toLowerCase();
  const visible = rows.filter((row) =>
    !needle || row.patient_id.toLowerCase().includes(needle) || row.sex.toLowerCase().startsWith(needle));
  const { key, dir } = state.sort;
  visible.sort((a, b) => {
    const x = a[key];
    const y = b[key];
    if (x === y) return a.patient_id.localeCompare(b.patient_id);
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * dir;
  });

  const head = el("tr", null, columns.map((column) => {
    const sorted = state.sort.key === column.key;
    return el("th", {
      scope: "col",
      className: [column.num ? "num" : "", column.narrow ? "hide-narrow" : ""].join(" ").trim(),
      "aria-sort": sorted ? (dir > 0 ? "ascending" : "descending") : "none",
    }, el("button", {
      type: "button",
      className: "sort",
      onclick: () => {
        state.sort = { key: column.key, dir: sorted ? -dir : column.key === "risk" || column.key === "flags" ? -1 : 1 };
        renderPatients();
      },
    }, column.label, el("span", { className: "sort-glyph", "aria-hidden": "true", text: sorted ? (dir > 0 ? "▲" : "▼") : "" })));
  }));

  const body = el("tbody");
  for (const row of visible) {
    const cells = {
      patient_id: el("td", null, el("button", {
        type: "button", className: "patient-open", text: row.patient_id,
        "aria-label": `Open patient ${row.patient_id}`,
        onclick: () => openPatient(row.patient_id, row.match),
      })),
      age: el("td", { className: "num", text: fmtAge(row.age) }),
      sex: el("td", { text: sexLabel(row.sex) }),
      matched: el("td", { className: "hide-narrow" },
        el("span", { className: "steps-matched" }, (row.match.matched_steps || []).map((s) =>
          el("span", { className: "pill", text: s.replaceAll("_", " ") })))),
      risk: el("td", null, row.analysis && row.analysis.risk ? riskMeter(row.analysis.risk) : el("span", { className: "muted", text: "-" })),
      flags: el("td", null, el("span", { className: "flags" },
        row.high > 0 && el("span", { className: "flag high", title: "results above the reference interval", text: `${row.high} high` }),
        row.low > 0 && el("span", { className: "flag low", title: "results below the reference interval", text: `${row.low} low` }),
        row.flags === 0 && el("span", { className: "muted", text: "None" }))),
      evidence: el("td", { className: "num hide-narrow", text: fmt(row.evidence) }),
    };
    put(body, el("tr", null, columns.map((column) => cells[column.key])));
  }
  if (!visible.length) {
    put(body, el("tr", { className: "empty-row" }, el("td", {
      colspan: String(columns.length),
      text: rows.length ? "No patient in this cohort matches the filter." : "No patient matched this question.",
    })));
  }

  put(view, 
    el("div", { className: "table-wrap" },
      el("table", null,
        el("caption", { className: "visually-hidden", text: "Patients in the cohort" }),
        el("thead", null, head),
        body)),
    el("p", {
      className: "table-foot",
      text: visible.length === rows.length
        ? `${plural(rows.length, "patient", "patients")}. Select an id to see the record behind the match.`
        : `${visible.length} of ${rows.length} patients shown.`,
    }));
}

function riskMeter(risk) {
  const pct = Math.round(risk.score * 100);
  return el("span", { className: "risk", title: `${risk.model_name} ${risk.model_version}` },
    el("span", {
      className: `meter ${risk.band}`, role: "img",
      "aria-label": `risk ${pct} percent, ${risk.band}`,
    }, el("span", { style: undefined, "data-width": String(pct) })),
    el("span", { className: "risk-label", text: `${risk.band} ${pct}%` }));
}

// Inline widths cannot come from a style attribute under the CSP, so they are set
// through the CSSOM after the element exists.
function applyWidths(root) {
  for (const node of root.querySelectorAll("[data-width]")) {
    node.style.width = `${Math.max(0, Math.min(100, Number(node.dataset.width)))}%`;
  }
}

// ---------------------------------------------------------------------------- profile

const NUMERIC_FEATURES = {
  age_years: ["Age", "years", "Patient"],
  hba1c_latest: ["HbA1c, latest", "%", "Observation"],
  egfr_latest: ["eGFR, latest", "mL/min/1.73m²", "Observation"],
  systolic_bp_latest: ["Systolic blood pressure, latest", "mmHg", "Observation"],
  ldl_cholesterol_latest: ["LDL cholesterol, latest", "mg/dL", "Observation"],
  condition_count: ["Active conditions", "per patient", "Condition"],
  active_medication_count: ["Active medications", "per patient", "MedicationRequest"],
  encounter_count: ["Encounters", "per patient", "Encounter"],
};

const FLAG_FEATURES = {
  is_female: ["Female", "Patient"],
  has_type_2_diabetes: ["Type 2 diabetes", "Condition"],
  has_hypertension: ["Hypertension", "Condition"],
  has_chronic_kidney_disease: ["Chronic kidney disease", "Condition"],
  on_biguanide: ["On metformin", "MedicationRequest"],
  on_insulin: ["On insulin", "MedicationRequest"],
  on_sglt2_inhibitor: ["On an SGLT2 inhibitor", "MedicationRequest"],
  medication_change_recent: ["Recent medication change", "MedicationRequest"],
};

function renderProfile(response) {
  const view = clear($("view-profile"));
  const stats = (response.cohort && response.cohort.statistics) || {};
  // A feature is only described when the query read its source in full. A Condition
  // search restricted to diabetes codes says nothing about hypertension, so showing
  // "Hypertension 0%" there would be a fabricated finding. Observation-derived values
  // are safe either way: an unread lab has no value and drops out on its own.
  const steps = response.query_plan.steps;
  const fullyRead = (type) =>
    type === "Patient" || type === "Observation" ||
    steps.some((s) => s.resource_type === type && !s.params.some((p) => ["code", "_id"].includes(p.name)));
  const fetched = { has: fullyRead };
  const total = response.cohort ? response.cohort.total_patients : 0;

  const numeric = Object.entries(NUMERIC_FEATURES).filter(([name, [, , source]]) =>
    stats[name] && fetched.has(source));
  const flags = Object.entries(FLAG_FEATURES).filter(([name, [, source]]) =>
    stats.flags && stats.flags[name] && fetched.has(source));
  const unmeasured = [...Object.entries(NUMERIC_FEATURES), ...Object.entries(FLAG_FEATURES)]
    .filter(([, spec]) => !fetched.has(spec[spec.length - 1]))
    .map(([, spec]) => spec[0]);

  if (!total) {
    put(view, el("p", { className: "muted", text: "There is no cohort to describe." }));
    return;
  }

  const sections = [];
  if (numeric.length) {
    sections.push(el("section", null,
      el("h3", { text: "Distributions" }),
      el("p", {
        className: "profile-sub",
        text: "Each strip spans the lowest to the highest value; the band is the middle half and the dot the median.",
      }),
      numeric.map(([name, [label, unit]]) => distributionRow(label, unit, stats[name], total)),
      dataTable(numeric.map(([name, [label, unit]]) => [label, unit, stats[name]]))));
  }
  if (flags.length) {
    sections.push(el("section", null,
      el("h3", { text: "Share of the cohort" }),
      el("p", { className: "profile-sub", text: "Among patients for whom the answer is known." }),
      flags.map(([name, [label]]) => proportionRow(label, stats.flags[name]))));
  }
  if (unmeasured.length) {
    sections.push(el("p", {
      className: "profile-sub",
      text: `Not described, because this query did not read every record they depend on: ${unmeasured.join(", ")}.`,
    }));
  }
  put(view, el("div", { className: "profile" }, sections));
  applyWidths(view);
}

function distributionRow(label, unit, s, total) {
  if (s.max === s.min) {
    return el("div", { className: "dist" },
      el("p", { className: "dist-name" }, label, el("small", { text: `${unit}, n = ${s.n} of ${total}` })),
      el("p", { className: "muted", text: `Every patient has the same value` }),
      el("p", { className: "dist-value" }, fmt(s.median), el("small", { text: "all" })));
  }
  const width = 400;
  const pad = 6;
  const span = s.max - s.min || 1;
  const x = (v) => pad + ((v - s.min) / span) * (width - pad * 2);
  const hasIqr = s.p25 !== undefined && s.p75 !== undefined;
  const title = `${label}: median ${fmt(s.median)}, middle half ${fmt(s.p25)} to ${fmt(s.p75)}, range ${fmt(s.min)} to ${fmt(s.max)} (n = ${s.n})`;
  const chart = svg("svg", { viewBox: `0 0 ${width} 34`, preserveAspectRatio: "none", role: "img", "aria-label": title },
    svg("title", null, title),
    svg("line", { class: "axis-line", x1: x(s.min), x2: x(s.max), y1: 12, y2: 12 }),
    hasIqr && svg("rect", { class: "iqr", x: x(s.p25), y: 6, width: Math.max(2, x(s.p75) - x(s.p25)), height: 12, rx: 3 }),
    svg("circle", { class: "median", cx: x(s.median), cy: 12, r: 5 }),
    svg("text", { class: "tick-label", x: x(s.min), y: 32, "text-anchor": "start" }, fmt(s.min)),
    svg("text", { class: "tick-label", x: x(s.max), y: 32, "text-anchor": "end" }, fmt(s.max)));
  return el("div", { className: "dist" },
    el("p", { className: "dist-name" }, label, el("small", { text: `${unit}, n = ${s.n} of ${total}` })),
    chart,
    el("p", { className: "dist-value" }, fmt(s.median), el("small", { text: "median" })));
}

function proportionRow(label, f) {
  const pct = f.n ? (f.count / f.n) * 100 : 0;
  return el("div", { className: "prop" },
    el("p", { className: "dist-name", text: label }),
    el("span", { className: "meter", role: "img", "aria-label": `${label}: ${fmt(pct)} percent` },
      el("span", { "data-width": String(pct) })),
    el("p", { className: "dist-value" }, `${fmt(pct, 0)}%`, el("small", { text: `${f.count} of ${f.n}` })));
}

function dataTable(rows) {
  return el("details", { className: "data-table-toggle" },
    el("summary", { text: "Show these values as a table" }),
    el("div", { className: "table-wrap" }, el("table", null,
      el("thead", null, el("tr", null, ["Measure", "n", "Min", "25th", "Median", "75th", "Max"].map((h, i) =>
        el("th", { scope: "col", className: i ? "num" : "", text: h })))),
      el("tbody", null, rows.map(([label, unit, s]) => el("tr", null,
        el("th", { scope: "row", text: `${label} (${unit})` }),
        [s.n, s.min, s.p25, s.median, s.p75, s.max].map((v) => el("td", { className: "num", text: fmt(v) }))))))));
}

// ---------------------------------------------------------------------------- queries

const STAGE_LABELS = {
  planning: "Planning",
  retrieval: "Reading FHIR",
  cohort: "Combining steps",
  demographics: "Demographics",
  normalization: "Normalising",
  screening: "Screening results",
  features: "Features",
  analysis: "Analytics",
  narrative: "Written summary",
};

function renderQueries(response) {
  const view = clear($("view-queries"));
  const queries = response.fhir_queries || [];
  put(view, 
    el("p", { className: "profile-sub", text: "The exact read-only searches sent to the FHIR server, in order." }),
    el("ol", { className: "query-list" }, queries.map((query) =>
      el("li", { className: "query-item" },
        el("code", { text: query }),
        el("button", {
          type: "button", className: "copy", text: "Copy",
          onclick: async (event) => {
            try {
              await navigator.clipboard.writeText(query);
              event.target.textContent = "Copied";
            } catch (_) {
              event.target.textContent = "Select and copy";
            }
          },
        })))));

  const stages = (response.trace && response.trace.stages) || {};
  const longest = Math.max(1, ...Object.values(stages));
  put(view, el("section", { className: "timing" },
    el("h3", { text: "Where the time went" }),
    Object.entries(stages).map(([name, ms]) =>
      el("div", { className: "timing-row" },
        el("span", { text: STAGE_LABELS[name] || name }),
        el("span", { className: "meter", "aria-hidden": "true" }, el("span", { "data-width": String((ms / longest) * 100) })),
        el("span", { className: "num", text: `${fmt(ms, 0)} ms` })))));
  applyWidths(view);
}

// ---------------------------------------------------------------------------- tabs

const TAB_IDS = ["tab-patients", "tab-profile", "tab-queries"];

function selectTab(id, focus = true) {
  for (const tabId of TAB_IDS) {
    const tab = $(tabId);
    const selected = tabId === id;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    $(tab.getAttribute("aria-controls")).hidden = !selected;
  }
  if (focus) $(id).focus();
  $("filter").hidden = id !== "tab-patients";
}

function setupTabs() {
  for (const id of TAB_IDS) {
    const tab = $(id);
    tab.addEventListener("click", () => selectTab(id, false));
    tab.addEventListener("keydown", (event) => {
      const index = TAB_IDS.indexOf(id);
      if (event.key === "ArrowRight") selectTab(TAB_IDS[(index + 1) % TAB_IDS.length]);
      else if (event.key === "ArrowLeft") selectTab(TAB_IDS[(index + TAB_IDS.length - 1) % TAB_IDS.length]);
      else return;
      event.preventDefault();
    });
  }
  $("filter").addEventListener("input", (event) => {
    state.filter = event.target.value.trim();
    renderPatients();
    applyWidths($("view-patients"));
  });
}

// ---------------------------------------------------------------------------- export

const EXPORT_NAMES = { csv: "cohort.csv", group: "cohort-group.json", bundle: "cohort-bundle.json" };

function setupExport() {
  const menu = $("export-menu");
  for (const button of menu.querySelectorAll("button[data-format]")) {
    button.addEventListener("click", async () => {
      menu.open = false;
      if (!state.lastBody) return;
      const format = button.dataset.format;
      try {
        const response = await request(`/query/export?format=${encodeURIComponent(format)}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ ...state.lastBody, narrate: false }),
        });
        const url = URL.createObjectURL(await response.blob());
        const link = el("a", { href: url, download: EXPORT_NAMES[format] || "cohort" });
        document.body.append(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        toast(`Downloaded ${EXPORT_NAMES[format]}`);
      } catch (err) {
        toast(`Export failed: ${err.lines ? err.lines[0] : err.message}`);
      }
    });
  }
  document.addEventListener("click", (event) => {
    if (menu.open && !menu.contains(event.target)) menu.open = false;
  });
  menu.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && menu.open) {
      menu.open = false;
      menu.querySelector("summary").focus();
    }
  });
}

// ---------------------------------------------------------------------------- patient drawer

function setupDrawer() {
  const drawer = $("patient-drawer");
  $("drawer-close").addEventListener("click", () => drawer.close());
  drawer.addEventListener("close", () => setUrlParam("patient", null));
  drawer.addEventListener("click", (event) => {
    if (event.target === drawer) drawer.close();
  });
  $("patient-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("patient-id");
    if (!input.reportValidity()) return;
    openPatient(input.value.trim(), null);
  });
}

function setUrlParam(name, value) {
  const params = new URLSearchParams(location.search);
  if (value) params.set(name, value);
  else params.delete(name);
  const query = params.toString();
  try {
    history.replaceState(null, "", query ? `?${query}` : location.pathname);
  } catch (_) {
    /* sandboxed documents may refuse history changes */
  }
}

async function openPatient(patientId, match) {
  const drawer = $("patient-drawer");
  setUrlParam("patient", patientId);
  $("drawer-title").textContent = patientId;
  const body = clear($("drawer-body"));
  put(body, el("div", { className: "loading", "aria-hidden": "true" },
    el("div", { className: "loading-line" }), el("div", { className: "skeleton" }), el("div", { className: "skeleton short" })));
  if (!drawer.open) drawer.showModal();

  const asOf = (state.lastBody && state.lastBody.as_of) || $("as-of").value;
  const query = asOf ? `?as_of=${encodeURIComponent(asOf)}` : "";
  try {
    const analysis = await api(`/patient/${encodeURIComponent(patientId)}/analyze${query}`);
    renderPatient(body, analysis, match);
  } catch (err) {
    put(clear(body), el("div", { className: "callout error", role: "alert" },
      el("h3", { text: err.status === 404 ? "No record for this id" : "The record could not be read" }),
      el("ul", null, err.lines.map((line) => el("li", { text: line }))),
      err.status === 404 && el("p", { text: "Check the id, including letter case." }),
      err.requestId && el("p", { className: "request-id" }, "Request id ", el("code", { text: err.requestId }))));
  }
}

function renderPatient(body, a, match) {
  clear(body);
  put(body, el("section", null,
    el("dl", { className: "facts" },
      fact("Age", fmtAge(a.age_years)),
      fact("Sex", sexLabel(a.gender)),
      fact("Out of range", String((a.abnormal_labs || []).length)),
      fact("Evidence", String((a.evidence || []).length)))));

  if (match && (match.matched_steps || []).length) {
    put(body, el("section", null,
      el("h3", { text: "Why this patient is in the cohort" }),
      el("p", { text: `Found by ${match.matched_steps.map((s) => s.replaceAll("_", " ")).join(", ")}.` }),
      match.summary && el("p", { className: "muted", text: match.summary })));
  }

  if (a.risk) {
    const pct = Math.round(a.risk.score * 100);
    put(body, el("section", { className: "risk-panel" },
      el("h3", { text: "Deterioration risk (demonstration model)" }),
      el("p", null, el("strong", { text: `${a.risk.band.charAt(0).toUpperCase()}${a.risk.band.slice(1)}` }), `, score ${pct}%`),
      el("span", { className: `meter ${a.risk.band}`, role: "img", "aria-label": `risk ${pct} percent` },
        el("span", { "data-width": String(pct) })),
      a.risk.contributing_factors.length > 0 && el("ul", null, a.risk.contributing_factors.map((f) => el("li", { text: f }))),
      a.risk.missing_features.length > 0 && el("p", {
        className: "muted",
        text: `Computed without: ${a.risk.missing_features.map(humanFeature).join(", ")}.`,
      })));
  }

  const labs = a.abnormal_labs || [];
  put(body, el("section", null,
    el("h3", { text: "Results outside the reference interval" }),
    labs.length ? labs.map(labRow) : el("p", { className: "muted", text: "None in the screening window." })));

  if ((a.data_gaps || []).length) {
    put(body, el("section", null,
      el("h3", { text: "Missing from the record" }),
      el("div", { className: "gaps" }, a.data_gaps.map((g) => el("span", { className: "pill", text: humanFeature(g) })))));
  }

  const evidence = a.evidence || [];
  put(body, el("section", null,
    el("h3", { text: "Evidence" }),
    evidence.length
      ? el("ul", { className: "evidence-list" }, evidence.map((e) => el("li", null,
        el("span", null, e.display || e.concept || e.resource_type,
          e.effective && el("span", { className: "muted", text: ` on ${fmtDate(e.effective)}` })),
        el("span", { className: "val", text: e.value || "" }),
        el("span", { className: "ref", text: `${e.resource_type}/${e.resource_id}` }))))
      : el("p", { className: "muted", text: "No supporting resources." })));

  const features = Object.entries(a.features || {}).filter(([, v]) => v !== null && v !== undefined);
  if (features.length) {
    put(body, el("details", { className: "features-toggle" },
      el("summary", { text: `All computed features (${features.length})` }),
      el("div", { className: "features-grid" }, features.map(([k, v]) =>
        el("div", null, el("code", { text: k }), el("span", { text: typeof v === "boolean" ? (v ? "yes" : "no") : fmt(v, 2) }))))));
  }

  put(body, el("p", {
    className: "drawer-disclaimer",
    text: (a.risk && a.risk.disclaimer) || "Research demonstration on synthetic data. Not for clinical use.",
  }));
  applyWidths(body);
}

function fact(label, value) {
  return el("div", null, el("dt", { text: label }), el("dd", { text: value }));
}

function humanFeature(name) {
  const known = NUMERIC_FEATURES[name] || FLAG_FEATURES[name];
  if (known) return known[0];
  return name.replace(/_latest$/, "").replaceAll("_", " ");
}

function labRow(lab) {
  const direction = lab.flag.includes("high") ? "high" : "low";
  const critical = lab.flag.startsWith("critical");
  const low = lab.reference_low;
  const high = lab.reference_high;
  const value = lab.value;
  const anchors = [low, high, value].filter((v) => typeof v === "number");
  let min = Math.min(...anchors);
  let max = Math.max(...anchors);
  if (high === null || high === undefined) max = Math.max(max, (low || value) * 1.4);
  if (low === null || low === undefined) min = Math.min(min, 0);
  const pad = (max - min || 1) * 0.12;
  min -= pad;
  max += pad;
  const width = 400;
  const x = (v) => ((v - min) / (max - min)) * width;
  const bandStart = typeof low === "number" ? x(low) : 0;
  const bandEnd = typeof high === "number" ? x(high) : width;
  const range = [low, high].every((v) => typeof v === "number")
    ? `${fmt(low)} to ${fmt(high)}`
    : typeof low === "number" ? `at least ${fmt(low)}` : `at most ${fmt(high)}`;
  const label = `${lab.display || lab.concept}: ${fmt(value)} ${lab.unit || ""}, reference ${range}`;

  const chart = typeof value === "number" && svg("svg", {
    viewBox: `0 0 ${width} 30`, preserveAspectRatio: "none", role: "img", "aria-label": label,
  },
  svg("title", null, label),
  svg("line", { class: "track", x1: 0, x2: width, y1: 10, y2: 10 }),
  svg("rect", { class: "range-band", x: bandStart, y: 4, width: Math.max(2, bandEnd - bandStart), height: 12, rx: 2 }),
  typeof low === "number" && svg("line", { class: "range-edge", x1: x(low), x2: x(low), y1: 3, y2: 17 }),
  typeof high === "number" && svg("line", { class: "range-edge", x1: x(high), x2: x(high), y1: 3, y2: 17 }),
  svg("circle", { class: `value-dot ${direction}`, cx: x(value), cy: 10, r: 6 }),
  typeof low === "number" && svg("text", { class: "tick-label", x: x(low), y: 29, "text-anchor": "middle" }, fmt(low)),
  typeof high === "number" && svg("text", { class: "tick-label", x: x(high), y: 29, "text-anchor": "middle" }, fmt(high)));

  return el("div", { className: "lab" },
    el("div", null,
      el("p", { className: "lab-name", text: lab.display || lab.concept }),
      el("p", { className: "lab-when", text: `${fmtDate(lab.effective)}, reference ${range} ${lab.unit || ""}` })),
    el("div", { className: "lab-value" },
      `${fmt(value)} ${lab.unit || ""} `,
      el("span", { className: `flag ${direction}`, text: `${critical ? "Critical " : ""}${direction}` })),
    chart);
}

// ---------------------------------------------------------------------------- misc

let toastTimer = null;

function toast(message) {
  const node = $("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.hidden = true;
  }, 3200);
}

function setupShortcuts() {
  document.addEventListener("keydown", (event) => {
    if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target.isContentEditable) return;
    if (document.querySelector("dialog[open]")) return;
    event.preventDefault();
    $("question").focus();
  });
  if (/Mac|iPhone|iPad/.test(navigator.platform || "")) {
    const hint = $("shortcut-hint");
    const keys = hint.querySelectorAll("kbd");
    if (keys[1]) keys[1].textContent = "⌘";
  }
}

function restoreFromUrl() {
  const params = new URLSearchParams(location.search);
  const question = params.get("q");
  const patient = params.get("patient");
  if (patient && /^[A-Za-z0-9\-.]{1,64}$/.test(patient)) openPatient(patient, null);
  if (!question) return;
  $("question").value = question;
  const asOf = params.get("as_of");
  if (asOf && /^\d{4}-\d{2}-\d{2}$/.test(asOf)) $("as-of").value = asOf;
  $("query-form").requestSubmit();
}

// Widths inside freshly rendered patient rows are applied after each render.
const observer = new MutationObserver((mutations) => {
  for (const mutation of mutations) {
    for (const node of mutation.addedNodes) if (node instanceof Element) applyWidths(node);
  }
});

document.addEventListener("DOMContentLoaded", () => {
  observer.observe(document.body, { childList: true, subtree: true });
  setupAccess();
  setupQueryForm();
  setupTabs();
  setupExport();
  setupDrawer();
  setupShortcuts();
  renderRecent();
  loadStatus();
  loadLibrary();
  restoreFromUrl();
});
