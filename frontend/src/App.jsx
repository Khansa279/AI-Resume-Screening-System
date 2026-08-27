import { useState } from 'react'
import JobDescriptionForm from './components/JobDescriptionForm'
import ResumeUploader from './components/ResumeUploader'
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
      <header className="app-header">
        <h1 className="app-title">AI Resume Screening System</h1>
        <p className="app-subtitle">
          Upload a job description and candidate resumes to generate a ranked shortlist.
        </p>
      </header>

      <main className="app-main">
        <form className="screening-card" onSubmit={handleSubmit}>
          <JobDescriptionForm
            jobTitle={jobTitle}
            onJobTitleChange={setJobTitle}
            jobDescriptionText={jobDescriptionText}
            onJobDescriptionTextChange={setJobDescriptionText}
            jobDescriptionFile={jobDescriptionFile}
            onJobDescriptionFileChange={handleJobDescriptionFileChange}
          />

          <div className="divider" />

          <ResumeUploader
            files={resumeFiles}
            onFilesAdded={handleResumesAdded}
            onFileRemoved={handleResumeRemoved}
          />

          <div className="submit-row">
            {statusMessage && <span className="status-message">{statusMessage}</span>}
            <button type="submit" className="primary-button" disabled={!canSubmit}>
              Screen Resumes
            </button>
          </div>
        </form>
      </main>
    </div>
  )
}

export default App
