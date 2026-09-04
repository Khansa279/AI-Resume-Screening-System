/**
 * API service layer for the AI Resume Screening System.
 *
 * Centralizes all HTTP calls to the existing FastAPI backend (src/api/).
 * No backend logic lives here -- this only wraps fetch() calls with a
 * consistent base URL, error handling, and payload shapes so components
 * never construct URLs or parse responses directly.
 *
 * Endpoints wrapped here (see src/api/routes/*.py):
 *   GET  /health
 *   GET  /jobs                          (screening history, newest first)
 *   POST /jobs                          (multipart/form-data)
 *   GET  /jobs/{position_id}
 *   POST /screening/{position_id}/run   (multipart/form-data)
 *   GET  /screening/{position_id}/results
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

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
    return detail.map((entry) => entry.msg || JSON.stringify(entry)).join(' ')
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
 * POST /jobs (multipart/form-data): title (required), organization and
 * department (optional -- omit to fall back to the backend's default
 * bucket), and exactly one of jdText / jdFile.
 *
 * Returns JobCreateResponse: { position_id, organization, department,
 * title, jd_version }.
 */
export function createJob({ title, organization, department, jdText, jdFile }) {
  const formData = new FormData()
  formData.append('title', title)
  if (organization) formData.append('organization', organization)
  if (department) formData.append('department', department)
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
 * Fetch a single position's organization/title context.
 * Returns JobSummary: { position_id, title, organization, department,
 * current_jd_version }.
 */
export function getJob(positionId) {
  return request(`/jobs/${positionId}`, { method: 'GET' })
}

/**
 * List every position ever screened (or set up) via this workspace,
 * newest activity first, with organization/position context and a
 * summary of its most recent ranking if one exists.
 *
 * Returns JobHistoryEntry[]: { position_id, title, organization,
 * department, jd_version, status, ranking_id, candidates_screened,
 * generated_at, top_candidate_name, top_candidate_score }.
 */
export function listJobHistory() {
  return request('/jobs', { method: 'GET' })
}

/**
 * Run screening for a position against its current job description,
 * uploading one or more resume files in the same request.
 *
 * POST /screening/{position_id}/run (multipart/form-data): resumes (one
 * or more files, field name "resumes").
 *
 * Returns ScreeningResponse: { position_id, ranking_id,
 * candidates_screened, results: [...], title, organization, department,
 * generated_at }.
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
  getJob,
  listJobHistory,
  runScreening,
  getResults,
}
