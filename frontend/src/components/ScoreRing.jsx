/**
 * Small radial meter used to show a candidate's match percentage. Reuses
 * the same "instrument" visual language as the hero signal bars and the
 * breakdown meters in CandidateDetail, so the whole app reads as one
 * consistent measurement system rather than mixed widget styles.
 */
import './ScoreRing.css'

function scoreTone(score) {
  if (score >= 0.75) return 'high'
  if (score >= 0.5) return 'medium'
  return 'low'
}

function ScoreRing({ score, size = 56 }) {
  const radius = (size - 6) / 2
  const circumference = 2 * Math.PI * radius
  const pct = Math.max(0, Math.min(1, score))
  const dash = circumference * pct
  const tone = scoreTone(pct)

  return (
    <div className={`score-ring score-ring--${tone}`} style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
        <circle
          className="score-ring__track"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth="4"
        />
        <circle
          className="score-ring__value"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          strokeWidth="4"
          strokeDasharray={`${dash} ${circumference}`}
          strokeLinecap="round"
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
      </svg>
      <span className="score-ring__label">{Math.round(pct * 100)}</span>
    </div>
  )
}

export default ScoreRing
