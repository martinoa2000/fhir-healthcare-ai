# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the fhir-healthcare-ai API.
#
# Stage 1 resolves and installs everything into a self-contained virtualenv;
# stage 2 keeps only that virtualenv, so no compilers, pip caches or build
# metadata survive into the published image.

# ---------------------------------------------------------------------------
# Stage 1 - builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Everything is installed into a relocatable venv that the runtime stage copies
# verbatim. `--copies` avoids symlinks back into this stage's interpreter.
RUN python -m venv --copies /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

# Dependency layer. Only the build metadata is copied here, so editing anything
# under src/ does not invalidate the (slow) resolve+download of numpy, pandas
# and scikit-learn. A placeholder package is enough for hatchling to build the
# wheel whose dependencies we actually want installed.
COPY pyproject.toml README.md ./
RUN mkdir -p src/fhir_healthcare_ai \
    && touch src/fhir_healthcare_ai/__init__.py \
    && pip install --no-cache-dir .

# Source layer. Reinstalling with --no-deps replaces the placeholder with the
# real package and takes about a second.
COPY src ./src
RUN pip install --no-cache-dir --no-deps --force-reinstall .

# ---------------------------------------------------------------------------
# Stage 2 - runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="fhir-healthcare-ai" \
      org.opencontainers.image.description="A governed AI layer over interoperable clinical data (HL7 FHIR R4)" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/your-org/fhir-healthcare-ai"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}"

# Fixed uid/gid so bind-mounted host directories have predictable ownership.
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /usr/sbin/nologin appuser

COPY --from=builder --chown=root:root /opt/venv /opt/venv

WORKDIR /app

# Writable location for generated bundles and the JSONL audit log; the image
# itself stays read-only-friendly (`docker run --read-only` works if /app/data
# is a volume).
RUN install -d -o appuser -g appuser /app/data

USER appuser

EXPOSE 8000

# Liveness only: /health/live must not touch the upstream FHIR server, otherwise
# Docker would restart a perfectly healthy API whenever HAPI hiccups. Probing
# with stdlib urllib keeps curl out of the runtime image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "fhir_healthcare_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
