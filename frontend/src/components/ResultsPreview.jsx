import CandidateCard from './CandidateCard'
import SignalBars from './SignalBars'
import './ResultsPreview.css'

/**
 * Maps one ScreeningCandidateResult (src/api/schemas.py) into the shape
 * CandidateCard expects. As of the F-08 backend fix the API now returns
 * contact info, matched skills / gaps, a skill/experience/role-relevance
 * breakdown, and a free-text explanation -- all mapped through here.
 * Anything still missing for a given candidate (e.g. no email on file)
 * is left undefined so CandidateCard continues to render its existing
 * "unavailable" fallback for that piece rather than fabricated data.
 */
function toCandidateCardProps(result, index) {
  const hasBreakdown =
    result.skill_match_score != null ||
    result.experience_score != null ||
    result.role_relevance != null

  return {
    id: result.screening_id ?? `${result.resume_id}-${index}`,
    rank: index + 1,
    name: result.candidate_name || 'Unknown',
    matchScore: result.match_score,
    recommendation: result.recommendation,
    requiresHuman: result.requires_human,
    confidence: result.confidence,
    error: result.error || null,
    email: result.candidate_email || undefined,
    phone: result.candidate_phone || undefined,
    explanation: result.why_summary || undefined,
    matchedSkills: result.matching_skills && result.matching_skills.length > 0
      ? result.matching_skills
      : undefined,
    skillGaps: result.skill_gaps && result.skill_gaps.length > 0
      ? result.skill_gaps
      : undefined,
    breakdown: hasBreakdown
      ? {
          skillMatch: result.skill_match_score ?? 0,
          experience: result.experience_score ?? 0,
          roleRelevance: result.role_relevance ?? 0,
        }
      : undefined,
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
