# fhir-healthcare-ai

Capa de IA gobernada sobre datos clínicos HL7 FHIR R4: un modelo local convierte una pregunta en un plan de consulta validado y devuelve los pacientes con la evidencia de cada uno.

## Seguimiento

- Estado, prioridad y roadmap viven en el dashboard git (~/git/.project-dashboard, http://localhost:1403); no los dupliques aquí.
- Al terminar una sesión de trabajo, actualiza NEXT.md con el siguiente paso: el dashboard lo lee.

## Stack

- Python 3.11+, FastAPI, pydantic; UI web estática (HTML/CSS/JS) sin build ni CDN.
- HL7 FHIR R4 (HAPI FHIR en Docker, o servidor en memoria para tests y demo).
- Inferencia solo local: servidor vLLM (Qwen3.5-9B); planificador determinista `mock` como alternativa.
- ruff, mypy, pytest; Docker Compose para el stack completo.

## Comandos

- Instalar: `make install`
- Arrancar: `make demo` (API y UI en http://127.0.0.1:8000, 120 pacientes sintéticos en memoria); `make run` con recarga; `make up` para el stack completo con Docker
- Tests: `make test` (unitarios); `make up && make test-integration` contra HAPI; `make cov` con cobertura
- Lint: `make lint` (ruff) y `make typecheck` (mypy); `make check` ejecuta lint, tipos, tests y benchmark
- Benchmark: `make bench`

## Convenciones

- No commitear datos, checkpoints ni secretos: `.env`, `data/`, `*.ndjson`, `*.jsonl`, `benchmark-results/` y pesos de modelos están en `.gitignore`. Nunca datos de pacientes reales; todo es sintético y reproducible con `fhir-ai-seed`.
- El modelo solo emite un `QueryPlan`; las URLs las construye únicamente el query builder y solo con GET de recursos de la allowlist (`fhir/allowlist.py`, `fhir/validator.py`).
- Los conceptos se nombran por clave y se expanden a códigos en `terminology.py`; el modelo nunca escribe códigos LOINC/SNOMED.
- Cambios notables en `CHANGELOG.md`; ver `CONTRIBUTING.md`.

## Contexto

- Flujo fijo en Python, no un agente: el modelo se llama como máximo dos veces (planificar y, opcionalmente, narrar resultados ya calculados). Si no puede expresar la pregunta completa, la rechaza con motivo.
- Seguridad: allowlist, solo lectura, límites de paginación, autenticación por clave, rate limit y auditoría JSONL.
- Benchmark con el planificador de reglas (grupo de control): 19/19 cohortes exactas, 3/3 rechazos, 0 violaciones; no demuestra comprensión, solo que el pipeline ejecuta bien un plan correcto. Falta medir el modelo local (`fhir-ai-bench --provider vllm`).
- Limitaciones: el planificador de reglas solo reconoce formas de pregunta fijas; el modelo de riesgo es una demostración, no validado clínicamente; `value-quantity` compara en una sola unidad (resultados en otra unidad, p. ej. HbA1c en mmol/mol, quedan invisibles); "último año/mes" = 365/30 días; fechas de nacimiento solo con año se comparan como rangos FHIR.
- Solo demostración de investigación/ingeniería: no usar para decisiones clínicas.
- Versión actual: 0.2.0 (terminado); en Unreleased solo hay mejoras de README.
