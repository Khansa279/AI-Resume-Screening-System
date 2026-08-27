import { useState } from 'react'
import JobDescriptionForm from './components/JobDescriptionForm'
import ResumeUploader from './components/ResumeUploader'
import ResultsPreview from './components/ResultsPreview'
import SignalBars from './components/SignalBars'
import './App.css'

function App() {
  const [jobTitle, setJobTitle] = useState('')
  const [jobDescriptionText, setJobDescriptionText] = useState('')
  const [jobDescriptionFile, setJobDescriptionFile] = useState(null)
  const [resumeFiles, setResumeFiles] = useState([])
  const [statusMessage, setStatusMessage] = useState(null)

  const handleJobDescriptionFileChange = (file) => {
    setJobDescriptionFile(file)
    if (file) {
      setJobDescriptionText('')
    }
  }

  const handleResumesAdded = (newFiles) => {
    setResumeFiles((prev) => [...prev, ...newFiles])
  }

  const handleResumeRemoved = (index) => {
    setResumeFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const hasJobDescription = Boolean(jobDescriptionText.trim() || jobDescriptionFile)
  const canSubmit = jobTitle.trim() && hasJobDescription && resumeFiles.length > 0

  const handleSubmit = (event) => {
    event.preventDefault()
    if (!canSubmit) return

    // Screening submission (API wiring for run/upload/results comes in a
    // later chunk) -- this confirms the collected inputs are ready to
    // send to the existing FastAPI endpoints in src/services/api.js.
    setStatusMessage(
      `Ready to screen ${resumeFiles.length} resume(s) for "${jobTitle.trim()}".`
    )
  }

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">
        Skip to workspace
      </a>

      <header className="site-header">
        <div className="site-header__brand">
          <span className="site-header__mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span className="site-header__name">ScreenIQ</span>
        </div>
        <span className="site-header__status">
          <span className="site-header__status-dot" aria-hidden="true" />
          Local workspace
        </span>
      </header>

      <section className="hero">
        <div className="hero__copy">
          <span className="eyebrow">AI-assisted screening</span>
          <h1 className="hero__title">
            Screen smarter.
            <br />
            Find the strongest candidates.
          </h1>
          <p className="hero__subtitle">
            Drop in a job description and a stack of resumes. ScreenIQ reads every
            candidate against the role and hands back a ranked, explainable shortlist
            &mdash; skills matched, gaps flagged, experience weighed.
          </p>
        </div>
        <div className="hero__signal" aria-hidden="true">
          <SignalBars />
          <span className="hero__signal-label">reading signal</span>
        </div>
      </section>

      <main className="app-main" id="workspace">
        <form className="screening-card" onSubmit={handleSubmit}>
          <div className="screening-card__head">
            <span className="eyebrow">Step 1</span>
            <h2 className="screening-card__title">Set up this screening</h2>
          </div>

          <div className="screening-card__columns">
            <JobDescriptionForm
              jobTitle={jobTitle}
              onJobTitleChange={setJobTitle}
              jobDescriptionText={jobDescriptionText}
              onJobDescriptionTextChange={setJobDescriptionText}
              jobDescriptionFile={jobDescriptionFile}
              onJobDescriptionFileChange={handleJobDescriptionFileChange}
            />

            <div className="column-divider" aria-hidden="true" />

            <ResumeUploader
              files={resumeFiles}
              onFilesAdded={handleResumesAdded}
              onFileRemoved={handleResumeRemoved}
            />
          </div>

          <div className="submit-row">
            {statusMessage && <span className="status-message">{statusMessage}</span>}
            <button type="submit" className="primary-button" disabled={!canSubmit}>
              Screen Resumes
            </button>
          </div>
        </form>

        <ResultsPreview />
      </main>

      <footer className="site-footer">
        <span>ScreenIQ workspace</span>
        <span className="site-footer__dot" aria-hidden="true">
          ·
        </span>
        <span>Results shown above use placeholder data</span>
      </footer>
    </div>
  )
}

export default App
