import { Route, Routes } from 'react-router-dom'
import AppShell from './components/AppShell'
import NewScreeningPage from './pages/NewScreeningPage'
import HistoryPage from './pages/HistoryPage'
import ResultsPage from './pages/ResultsPage'
import './App.css'

function App() {
  return (
    <AppShell>
      <a className="skip-link" href="#workspace">
        Skip to workspace
      </a>
      <Routes>
        <Route path="/" element={<NewScreeningPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/results/:positionId" element={<ResultsPage />} />
      </Routes>
      <footer className="site-footer">
        <div className="site-footer__inner">
          <span>ScreenIQ workspace</span>
          <span className="site-footer__dot" aria-hidden="true">
            ·
          </span>
          <span>Every screening you run is saved to your history automatically</span>
        </div>
      </footer>
    </AppShell>
  )
}

export default App
