import { NavLink } from 'react-router-dom'
import './AppShell.css'

/**
 * Application chrome shared by every route: the sticky top navigation
 * (brand + primary sections) plus the page body outlet. Keeping this
 * separate from the page content is what lets organization/position
 * context and the "New screening" workflow live on different routes
 * while the person never loses their place in the product.
 */
function AppShell({ children }) {
  return (
    <div className="app-shell">
      <header className="top-nav">
        <div className="top-nav__inner">
          <NavLink to="/" className="top-nav__brand" end>
            <span className="top-nav__mark" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
            <span className="top-nav__name">ScreenIQ</span>
          </NavLink>

          <nav className="top-nav__links" aria-label="Primary">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `top-nav__link ${isActive ? 'top-nav__link--active' : ''}`
              }
            >
              <span aria-hidden="true">＋</span>
              <span className="top-nav__link-label">New screening</span>
            </NavLink>
            <NavLink
              to="/history"
              className={({ isActive }) =>
                `top-nav__link ${isActive ? 'top-nav__link--active' : ''}`
              }
            >
              <span aria-hidden="true">☰</span>
              <span className="top-nav__link-label">Screening history</span>
            </NavLink>
          </nav>

          <span className="top-nav__status">
            <span className="top-nav__status-dot" aria-hidden="true" />
            Local workspace
          </span>
        </div>
      </header>

      <main className="app-body">{children}</main>
    </div>
  )
}

export default AppShell
