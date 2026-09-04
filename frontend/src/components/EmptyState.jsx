import { Link } from 'react-router-dom'
import './EmptyState.css'

/**
 * Reusable empty/guiding state. `actionTo` + `actionLabel` render a link
 * styled as a button when the empty state should point somewhere (e.g.
 * "no screenings yet -> start one"); omit them for a purely informational
 * empty state.
 */
function EmptyState({ icon = '✦', title, body, actionTo, actionLabel }) {
  return (
    <div className="empty-state">
      <span className="empty-state__icon" aria-hidden="true">
        {icon}
      </span>
      <p className="empty-state__title">{title}</p>
      {body && <p className="empty-state__body">{body}</p>}
      {actionTo && actionLabel && (
        <Link to={actionTo} className="primary-button empty-state__action">
          {actionLabel}
        </Link>
      )}
    </div>
  )
}

export default EmptyState
