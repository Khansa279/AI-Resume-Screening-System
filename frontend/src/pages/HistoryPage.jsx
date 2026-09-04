import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import EmptyState from '../components/EmptyState'
import SignalBars from '../components/SignalBars'
import { listJobHistory } from '../services/api'
import './HistoryPage.css'

function formatDate(value) {
  if (!value) return null
  try {
    return new Date(value).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
  } catch {
    return null
  }
}

function HistoryRow({ entry }) {
  const isScreened = entry.status === 'screened'
  return (
    <Link
      className="history-row"
      to={isScreened ? `/results/${entry.position_id}` : '#'}
      onClick={(e) => {
        if (!isScreened) e.preventDefault()
      }}
      aria-disabled={!isScreened}
    >
      <div className="history-row__top">
        <div className="history-row__role">
          <span className="history-row__org">
            {entry.organization}
            {entry.department ? ` · ${entry.department}` : ''}
          </span>
          <span className="history-row__title">{entry.title}</span>
        </div>
        <span
          className={`history-row__status history-row__status--${isScreened ? 'screened' : 'draft'}`}
        >
          {isScreened ? 'Screened' : 'Awaiting resumes'}
        </span>
      </div>

      <div className="history-row__meta">
        <span className="history-row__meta-item">
          <span className="history-row__meta-label">Candidates</span>
          <span className="history-row__meta-value">{entry.candidates_screened}</span>
        </span>
        {entry.jd_version && (
          <span className="history-row__meta-item">
            <span className="history-row__meta-label">JD version</span>
            <span className="history-row__meta-value">v{entry.jd_version}</span>
          </span>
        )}
        {formatDate(entry.generated_at) && (
          <span className="history-row__meta-item">
            <span className="history-row__meta-label">Last run</span>
            <span className="history-row__meta-value">{formatDate(entry.generated_at)}</span>
          </span>
        )}
      </div>

      {isScreened && entry.top_candidate_name && (
        <div className="history-row__top-candidate">
          <span>Top candidate:</span>
          <strong>{entry.top_candidate_name}</strong>
          {typeof entry.top_candidate_score === 'number' && (
            <span className="history-row__score">
              {Math.round(entry.top_candidate_score * 100)}%
            </span>
          )}
        </div>
      )}
    </Link>
  )
}

function HistoryPage() {
  const [entries, setEntries] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setIsLoading(true)
    setError(null)
    listJobHistory()
      .then((data) => {
        if (!cancelled) setEntries(data || [])
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || 'Could not load screening history.')
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="history-page">
      <div className="history-page__head">
        <span className="eyebrow">Screening history</span>
        <h1>Every role you&rsquo;ve set up</h1>
        <p>
          Organization, position, candidate counts, and the strongest match found so far
          &mdash; open any past screening to see the full ranked shortlist again.
        </p>
      </div>

      {isLoading && (
        <div className="history-page__status" role="status">
          <SignalBars />
          <p className="history-page__status-text">Loading your screening history&hellip;</p>
        </div>
      )}

      {!isLoading && error && (
        <div className="history-page__status history-page__status--error" role="alert">
          <p className="history-page__status-text">{error}</p>
        </div>
      )}

      {!isLoading && !error && entries.length === 0 && (
        <EmptyState
          icon="⌁"
          title="No screenings yet"
          body="Once you set up a role and screen a few resumes, every session will show up here so you can revisit it any time."
          actionTo="/"
          actionLabel="Start your first screening"
        />
      )}

      {!isLoading && !error && entries.length > 0 && (
        <div className="history-list">
          {entries.map((entry) => (
            <HistoryRow key={entry.position_id} entry={entry} />
          ))}
        </div>
      )}
    </div>
  )
}

export default HistoryPage
