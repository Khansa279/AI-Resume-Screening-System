import './OrgPositionHeader.css'

function formatDate(value) {
  if (!value) return null
  try {
    return new Date(value).toLocaleString(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    })
  } catch {
    return null
  }
}

/**
 * Persistent organization/position context banner. Shown above results
 * (freshly run or reopened from history) so the person never loses
 * track of which role a ranked shortlist belongs to.
 */
function OrgPositionHeader({ title, organization, department, candidatesScreened, generatedAt, status }) {
  const when = formatDate(generatedAt)

  return (
    <section className="role-header">
      <div className="role-header__main">
        {(organization || department) && (
          <div className="role-header__org">
            {organization && <span>{organization}</span>}
            {department && <span className="role-header__org-dept">{department}</span>}
          </div>
        )}
        <h2 className="role-header__title">{title || 'Untitled role'}</h2>

        <div className="role-header__meta">
          {typeof candidatesScreened === 'number' && (
            <div className="role-header__meta-item">
              <span className="role-header__meta-label">Candidates</span>
              <span className="role-header__meta-value">{candidatesScreened} screened</span>
            </div>
          )}
          {when && (
            <div className="role-header__meta-item">
              <span className="role-header__meta-label">Last run</span>
              <span className="role-header__meta-value">{when}</span>
            </div>
          )}
        </div>
      </div>

      {status && (
        <span className="role-header__badge">
          <span className="role-header__badge-dot" aria-hidden="true" />
          {status}
        </span>
      )}
    </section>
  )
}

export default OrgPositionHeader
