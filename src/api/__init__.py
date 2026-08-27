"""FastAPI application package.

This package is intentionally isolated from the existing screening
pipeline (src/agents, src/workflow, src/services, src/db): it only wires
HTTP requests/responses to the ALREADY-EXISTING, unmodified production
functions in src/services/screening_service.py and src/db/repository.py.
No resume parsing, scoring, job-relevance, or experience-evaluation
logic lives here or is duplicated here.
"""