import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import ResultsPreview from '../components/ResultsPreview'
import SignalBars from '../components/SignalBars'
import { getResults } from '../services/api'
import './ResultsPage.css'

/**
 * Reopens a previously-run screening (navigated to from Screening
 * History) without re-running anything -- GET /screening/{id}/results.
 */
function ResultsPage() {
  const { positionId } = useParams()
  const [response, setResponse] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setIsLoading(true)
    setError(null)
    getResults(positionId)
      .then((data) => {
        if (!cancelled) setResponse(data)
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || 'Could not load this screening.')
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [positionId])

  return (
    <div className="results-page">
      <Link to="/history" className="results-page__back">
        ← Back to screening history
      </Link>

      {isLoading && (
        <div className="results-page__status" role="status">
          <SignalBars />
          <p className="results-page__status-text">Loading this screening&hellip;</p>
        </div>
      )}

      {!isLoading && error && (
        <div className="results-page__status results-page__status--error" role="alert">
          <p className="results-page__status-text">{error}</p>
        </div>
      )}

      {!isLoading && !error && response && (
        <ResultsPreview
          jobTitle={response.title || ''}
          results={response.results || []}
          isLoading={false}
          error={null}
          screeningComplete
          meta={{
            title: response.title,
            organization: response.organization,
            department: response.department,
            generatedAt: response.generated_at,
            status: 'Previous run',
          }}
        />
      )}
    </div>
  )
}

export default ResultsPage
