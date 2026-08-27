import { useState } from 'react'
import ScoreRing from './ScoreRing'
import './CandidateCard.css'

const RECOMMENDATION_TONE = {
  'Proceed to interview': 'high',
  'Proceed to phone screening': 'medium',
  'Needs manual review': 'low',
  Reject: 'low',
}

function recommendationTone(recommendation) {
  return RECOMMENDATION_TONE[recommendation] || 'medium'
}

function initials(name) {
  return name
    .split(' ')
    .map((part) => part[0])
    .filter(Boolean)
    .slice(0, 2)
    .join('')
    .toUpperCase()
}

function ContactAction({ icon, label, value, href }) {
  return (
    <a className="contact-action" href={href} aria-label={label}>
      <span className="contact-action__icon" aria-hidden="true">
        {icon}
      </span>
      <span className="contact-action__tooltip" role="tooltip">
        <span className="contact-action__tooltip-label">{label}</span>
        <span className="contact-action__tooltip-value">{value}</span>
      </span>
    </a>
  )
}

function BreakdownMeter({ label, value }) {
  return (
    <div className="breakdown-meter">
      <div className="breakdown-meter__head">
        <span className="breakdown-meter__label">{label}</span>
        <span className="breakdown-meter__value">{Math.round(value * 100)}%</span>
      </div>
      <div className="breakdown-meter__track">
        <div className="breakdown-meter__fill" style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  )
}

function CandidateCard({ candidate }) {
  const [expanded, setExpanded] = useState(false)
  const tone = recommendationTone(candidate.recommendation)

  const toggle = () => setExpanded((v) => !v)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      toggle()
    }
  }

  return (
    <article className={`candidate-card ${expanded ? 'candidate-card--expanded' : ''}`}>
      <div
        className="candidate-card__summary"
        role="button"
        tabIndex={0}
        onClick={toggle}
        onKeyDown={handleKeyDown}
        aria-expanded={expanded}
      >
        <span className="candidate-card__rank">{String(candidate.rank).padStart(2, '0')}</span>

        <ScoreRing score={candidate.matchScore} />

        <span className="candidate-card__identity">
          <span className="candidate-card__avatar" aria-hidden="true">
            {initials(candidate.name)}
          </span>
          <span className="candidate-card__name-block">
            <span className="candidate-card__name">{candidate.name}</span>
            <span className="candidate-card__explanation">{candidate.explanation}</span>
          </span>
        </span>

        <span className={`recommendation-pill recommendation-pill--${tone}`}>
          {candidate.recommendation}
        </span>

        <span className="candidate-card__skills">
          {candidate.matchedSkills.slice(0, 4).map((skill) => (
            <span key={skill} className="skill-chip">
              {skill}
            </span>
          ))}
          {candidate.matchedSkills.length > 4 && (
            <span className="skill-chip skill-chip--more">+{candidate.matchedSkills.length - 4}</span>
          )}
        </span>

        <span className="candidate-card__contacts" onClick={(e) => e.stopPropagation()}>
          <ContactAction icon="✉" label="Email" value={candidate.email} href={`mailto:${candidate.email}`} />
          <ContactAction icon="☎" label="Phone" value={candidate.phone} href={`tel:${candidate.phone}`} />
          <ContactAction icon="📄" label="Resume" value={candidate.resumeFile} href="#" />
        </span>

        <span className="candidate-card__chevron" aria-hidden="true">
          {expanded ? '−' : '+'}
        </span>
      </div>

      {expanded && (
        <div className="candidate-card__detail">
          <div className="candidate-card__detail-grid">
            <section className="detail-block">
              <h4 className="detail-block__title">Score breakdown</h4>
              <BreakdownMeter label="Skill match" value={candidate.breakdown.skillMatch} />
              <BreakdownMeter label="Experience" value={candidate.breakdown.experience} />
              <BreakdownMeter label="Role relevance" value={candidate.breakdown.roleRelevance} />
            </section>

            <section className="detail-block">
              <h4 className="detail-block__title">Matched skills</h4>
              <div className="detail-block__chips">
                {candidate.matchedSkills.map((skill) => (
                  <span key={skill} className="skill-chip skill-chip--matched">
                    {skill}
                  </span>
                ))}
              </div>

              <h4 className="detail-block__title detail-block__title--spaced">Skill gaps</h4>
              <div className="detail-block__chips">
                {candidate.skillGaps.length > 0 ? (
                  candidate.skillGaps.map((skill) => (
                    <span key={skill} className="skill-chip skill-chip--gap">
                      {skill}
                    </span>
                  ))
                ) : (
                  <span className="detail-block__empty">No notable gaps identified</span>
                )}
              </div>
            </section>

            <section className="detail-block detail-block--wide">
              <h4 className="detail-block__title">AI explanation</h4>
              <p className="detail-block__paragraph">{candidate.explanation}</p>
            </section>
          </div>
        </div>
      )}
    </article>
  )
}

export default CandidateCard
