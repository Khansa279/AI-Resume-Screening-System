import './SignalBars.css'

const HEIGHTS = [6, 14, 9, 20, 12, 17, 8]

function SignalBars({ className = '' }) {
  return (
    <div className={`signal-bars ${className}`.trim()} aria-hidden="true">
      {HEIGHTS.map((h, i) => (
        <span key={i} className="signal-bars__bar" style={{ height: `${h}px` }} />
      ))}
    </div>
  )
}

export default SignalBars
