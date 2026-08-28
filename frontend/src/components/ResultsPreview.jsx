import CandidateCard from './CandidateCard'
import SignalBars from './SignalBars'
import './ResultsPreview.css'

/**
 * Maps one ScreeningCandidateResult (src/api/schemas.py) into the shape
 * CandidateCard expects. Only fields the API actually returns are used --
 * anything CandidateCard can't be given (email, phone, matched skills,
 * score breakdown, etc.) is left undefined so CandidateCard renders an
 * "unavailable" state for that piece instead of fabricated data.
 */
function toCandidateCardProps(result, index) {
  return {
    id: result.screening_id ?? `${result.resume_id}-${index}`,
    rank: index + 1,
    name: result.candidate_name || 'Unknown',
    matchScore: result.match_score,
    recommendation: result.recommendation,
    requiresHuman: result.requires_human,
    confidence: result.confidence,
    error: result.error || null,
  }
}

function ResultsPreview({ jobTitle, results, isLoading, error, screeningComplete }) {
  const hasResults = screeningComplete && results && results.length > 0

  return (
    <section className="results-preview" aria-labelledby="results-preview-heading">
      <div className="results-preview__header">
        <div>
          <span className="eyebrow">
            {screeningComplete ? 'Results' : 'Candidate ranking'}
          </span>
          <h2 id="results-preview-heading" className="results-preview__title">
            Candidate ranking
          </h2>
          <p className="results-preview__subtitle">
            {screeningComplete
              ? `Ranked shortlist for "${jobTitle.trim() || 'this role'}", straight from the screening pipeline.`
              : 'Set up a job description and upload resumes above, then run a screening to see your ranked shortlist here.'}
          </p>
        </div>
        {hasResults && (
          <span className="results-preview__count">{results.length} candidates</span>
        )}
      </div>

      {isLoading && (
        <div className="results-preview__status" role="status">
          <SignalBars />
          <p className="results-preview__status-text">
            Screening resumes against the job description&hellip; this can take a moment per candidate.
          </p>
        </div>
      )}

      {!isLoading && error && (
        <div className="results-preview__status results-preview__status--error" role="alert">
          <p className="results-preview__status-text">
            Screening didn&rsquo;t complete: {error}
          </p>
        </div>
      )}

      {!isLoading && !error && !screeningComplete && (
        <div className="results-preview__status results-preview__status--empty">
          <p className="results-preview__status-text">
            No screening has been run yet. Your ranked candidates will appear here.
          </p>
        </div>
      )}

      {!isLoading && !error && screeningComplete && !hasResults && (
        <div className="results-preview__status results-preview__status--empty">
          <p className="results-preview__status-text">
            The screening finished, but no candidates could be ranked.
          </p>
        </div>
      )}

      {hasResults && (
        <div className="results-preview__list">
          {results.map((result, index) => (
            <CandidateCard
              key={result.screening_id ?? `${result.resume_id}-${index}`}
              candidate={toCandidateCardProps(result, index)}
            />
          ))}
        </div>
      )}
    </section>
  )
}

export default ResultsPreview
