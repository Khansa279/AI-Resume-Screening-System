/**
 * API service layer for the AI Resume Screening System.
 *
 * Centralizes all HTTP calls to the existing FastAPI backend (src/api/).
 * No backend logic lives here -- this only wraps fetch() calls with a
 * consistent base URL, error handling, and payload shapes so components
 * never construct URLs or parse responses directly.
 *
 * The backend base URL can be overridden via the VITE_API_BASE_URL env
 * var (e.g. in a .env file) -- defaults to the typical local FastAPI dev
 * server address.
 *
 * IMPORTANT: this file only wraps endpoints that actually exist in
 * src/api/routes/*.py:
 *   GET  /health
 *   POST /jobs                          (multipart/form-data)
 *   POST /screening/{position_id}/run   (multipart/form-data)
 *   GET  /screening/{position_id}/results
 * There is no separate resume-upload endpoint -- resumes are sent
 * directly as part of the /screening/{position_id}/run request.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

/**
 * Generic fetch wrapper: builds the full URL, handles JSON/FormData
 * bodies, and normalizes error handling so callers get either parsed
 * JSON or a thrown Error with a useful message.
 *
 * Every error response from the backend (see src/api/main.py's
 * exception handlers) has the shape { error, detail }, where `detail`
 * is either a string (HTTPException / ValueError cases) or a list of
 * validation error objects (422 RequestValidationError). This wrapper
 * normalizes both into a single human-readable string so callers never
 * need to know which failure mode produced it.
 */
async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, options)
  } catch {
    throw new Error(
      `Could not reach the backend at ${API_BASE_URL}. Is the FastAPI server running?`
    )
  }

  let data = null
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    data = await response.json().catch(() => null)
  }

  if (!response.ok) {
    throw new Error(extractErrorMessage(data, response.status))
  }

  return data
}

function extractErrorMessage(data, status) {
  const detail = data && data.detail
  if (typeof detail === 'string' && detail.trim()) {
    return detail
  }
  if (Array.isArray(detail) && detail.length > 0) {
    // FastAPI/Pydantic 422 validation errors: [{ loc, msg, type }, ...]
    return detail
      .map((entry) => entry.msg || JSON.stringify(entry))
      .join(' ')
  }
  if (data && typeof data.message === 'string' && data.message.trim()) {
    return data.message
  }
  return `Request failed with status ${status}.`
}

/** GET /health -- basic backend liveness check. */
export function checkHealth() {
  return request('/health', { method: 'GET' })
}

/**
 * Create (or version-update) a job description for a position.
 *
 * POST /jobs (multipart/form-data): title (required), and exactly one
 * of jdText / jdFile.
 *
 * Returns JobCreateResponse: { position_id, organization, department,
 * title, jd_version }.
 */
export function createJob({ title, jdText, jdFile }) {
  const formData = new FormData()
  formData.append('title', title)
  if (jdFile) {
    formData.append('jd_file', jdFile)
  } else {
    formData.append('jd_text', jdText || '')
  }

  return request('/jobs', {
    method: 'POST',
    body: formData,
  })
}

/**
 * Run screening for a position against its current job description,
 * uploading one or more resume files in the same request.
 *
 * POST /screening/{position_id}/run (multipart/form-data): resumes (one
 * or more files, field name "resumes").
 *
 * Returns ScreeningResponse: { position_id, ranking_id,
 * candidates_screened, results: [...] }.
 */
export function runScreening(positionId, resumeFiles) {
  const formData = new FormData()
  resumeFiles.forEach((file) => {
    formData.append('resumes', file)
  })

  return request(`/screening/${positionId}/run`, {
    method: 'POST',
    body: formData,
  })
}

/**
 * Retrieve the most recently computed ranking/results for a position,
 * without re-running screening.
 *
 * GET /screening/{position_id}/results -- returns the same
 * ScreeningResponse shape as runScreening.
 */
export function getResults(positionId) {
  return request(`/screening/${positionId}/results`, {
    method: 'GET',
  })
}

export default {
  checkHealth,
  createJob,
  runScreening,
  getResults,
}
