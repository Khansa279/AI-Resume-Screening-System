import CandidateCard from './CandidateCard'
import { mockCandidates, mockJobTitle } from '../data/mockCandidates'
import './ResultsPreview.css'

function ResultsPreview() {
  return (
    <section className="results-preview" aria-labelledby="results-preview-heading">
      <div className="results-preview__header">
        <div>
          <span className="eyebrow">Preview · sample data</span>
          <h2 id="results-preview-heading" className="results-preview__title">
            Candidate ranking
          </h2>
          <p className="results-preview__subtitle">
            This is what your ranked shortlist will look like for &ldquo;{mockJobTitle}&rdquo; once
            screening runs. Click a candidate to see the full breakdown.
          </p>
        </div>
        <span className="results-preview__count">{mockCandidates.length} candidates · mock</span>
      </div>

      <div className="results-preview__list">
        {mockCandidates.map((candidate) => (
          <CandidateCard key={candidate.id} candidate={candidate} />
        ))}
      </div>
    </section>
  )
}

export default ResultsPreview
