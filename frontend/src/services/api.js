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
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

/**
 * Generic fetch wrapper: builds the full URL, handles JSON/FormData
 * bodies, and normalizes error handling so callers get either parsed
 * JSON or a thrown Error with a useful message.
 */
async function request(path, options = {}) {
  const url = `${API_BASE_URL}${path}`
  const response = await fetch(url, options)

  let data = null
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    data = await response.json().catch(() => null)
  }

  if (!response.ok) {
    const message =
      (data && (data.detail || data.message)) ||
      `Request failed with status ${response.status}`
    throw new Error(message)
  }

  return data
}

/** GET /health -- basic backend liveness check. */
export function checkHealth() {
  return request('/health', { method: 'GET' })
}

/**
 * Create (or version) a job description for a position.
 * jobPayload shape is left flexible since it's owned by the backend --
 * expected to include at least { title, description }.
 */
export function createJobDescription(jobPayload) {
  return request('/job-descriptions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(jobPayload),
  })
}

/** Upload a single resume file for a given position. */
export function uploadResume(positionId, file) {
  const formData = new FormData()
  formData.append('file', file)

  return request(`/positions/${positionId}/resumes`, {
    method: 'POST',
    body: formData,
  })
}

/** Upload multiple resumes sequentially, returning all results (and any errors). */
export async function uploadResumes(positionId, files) {
  const results = []
  for (const file of files) {
    try {
      const result = await uploadResume(positionId, file)
      results.push({ file, success: true, result })
    } catch (error) {
      results.push({ file, success: false, error: error.message })
    }
  }
  return results
}

/** Kick off screening for a position against its current job description. */
export function runScreening(positionId) {
  return request(`/positions/${positionId}/screen`, {
    method: 'POST',
  })
}

/** Retrieve the stored ranking/results for a position. */
export function getResults(positionId) {
  return request(`/positions/${positionId}/results`, {
    method: 'GET',
  })
}

export default {
  checkHealth,
  createJobDescription,
  uploadResume,
  uploadResumes,
  runScreening,
  getResults,
}
