"""Service entrypoint module exposing FastAPI app at
src.movemaster.service.api:app

This re-exports the application defined in src.movemaster.api.app
so Docker CMD can reference a stable path.
"""
from ..api.app import app  # noqa: F401
