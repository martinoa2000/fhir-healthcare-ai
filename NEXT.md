# Siguiente paso

Ejecutar `fhir-ai-bench --provider vllm --output benchmark-results/` con el modelo local y comparar el resultado con el grupo de control del planificador de reglas.

## Backlog

- [ ] Publicar en el README el resultado del benchmark con el modelo vLLM
- [ ] Añadir un caso al benchmark donde un resultado esté solo en mmol/mol (limitación de unidades de `value-quantity`)
- [ ] Medir también el ranking del cohorte de alto riesgo, no solo su pertenencia
- [ ] Revisar las lecturas de "último mes/año" (30/365 días) frente a ventanas de calendario
- [ ] Cerrar la sección Unreleased del CHANGELOG con una release
