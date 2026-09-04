import CandidateCard from './CandidateCard'
import OrgPositionHeader from './OrgPositionHeader'
import EmptyState from './EmptyState'
import SignalBars from './SignalBars'
import './ResultsPreview.css'

/**
 * Maps one ScreeningCandidateResult (src/api/schemas.py) into the shape
 * CandidateCard expects.
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
    matchedSkills: result.matching_skills || [],
    skillGaps: result.skill_gaps || [],
    explanation: result.explanation || null,
    email: result.candidate_email || null,
    phone: result.candidate_phone || null,
  }
}

/**
 * `meta` (optional): { title, organization, department, generatedAt,
 * status } -- organization/position context that must stay visible
 * alongside the ranked list, whether these results were just produced
 * by a fresh run or reopened from history.
 */
function ResultsPreview({ jobTitle, results, isLoading, error, screeningComplete, meta }) {
  const hasResults = screeningComplete && results && results.length > 0
  const showHeader = hasResults && meta

  return (
    <section className="results-preview" aria-labelledby="results-preview-heading">
      {showHeader && (
        <OrgPositionHeader
          title={meta.title || jobTitle}
          organization={meta.organization}
          department={meta.department}
          candidatesScreened={results.length}
          generatedAt={meta.generatedAt}
          status={meta.status || 'Screened'}
        />
      )}

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
        <EmptyState
          icon="⌁"
          title="No screening has been run yet"
          body="Fill in the role details and drop in a few resumes above, then run a screening to see your ranked shortlist here."
        />
      )}

      {!isLoading && !error && screeningComplete && !hasResults && (
        <EmptyState
          icon="⚠"
          title="The screening finished, but no candidates could be ranked"
          body="Every uploaded resume failed to process. Check the files and try again."
        />
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
