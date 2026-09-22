"""HTTP surface: a thin FastAPI layer over :class:`PipelineOrchestrator`.

Nothing clinical is decided here. The routes validate input, bind a correlation id,
call the pipeline and translate its exceptions into status codes.
"""

from fhir_healthcare_ai.api.main import create_app

__all__ = ["create_app"]
