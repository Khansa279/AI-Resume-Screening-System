import { useRef, useState } from 'react'
import FileAttachment from './FileAttachment'
import { isAcceptedJobDescriptionFile } from '../utils/fileHelpers'
import './JobDescriptionForm.css'

/**
 * Job title + job description input.
 *
 * The job description can be provided either as typed text OR as an
 * uploaded .txt file -- the two are mutually exclusive: uploading a file
 * clears/disables the textarea, and clearing the file re-enables typing.
 */
function JobDescriptionForm({
  jobTitle,
  onJobTitleChange,
  jobDescriptionText,
  onJobDescriptionTextChange,
  jobDescriptionFile,
  onJobDescriptionFileChange,
}) {
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileSelected = (file) => {
    if (!file) return
    if (!isAcceptedJobDescriptionFile(file)) {
      window.alert('Please upload a .txt file for the job description.')
      return
    }
    onJobDescriptionFileChange(file)
  }

  const handleInputChange = (event) => {
    const file = event.target.files?.[0]
    handleFileSelected(file)
    event.target.value = ''
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragActive(false)
    const file = event.dataTransfer.files?.[0]
    handleFileSelected(file)
  }

  const handleRemoveFile = () => {
    onJobDescriptionFileChange(null)
  }

  return (
    <section className="jd-form">
      <div className="field-group">
        <label htmlFor="job-title" className="field-label">
          Job Title
        </label>
        <input
          id="job-title"
          type="text"
          className="text-input"
          placeholder="e.g. Backend Engineer - Python"
          value={jobTitle}
          onChange={(event) => onJobTitleChange(event.target.value)}
        />
      </div>

      <div className="field-group">
        <div className="field-label-row">
          <label htmlFor="job-description" className="field-label">
            Job Description
          </label>
          <span className="field-hint">Type it below, or upload a .txt file</span>
        </div>

        {jobDescriptionFile ? (
          <div className="jd-file-preview">
            <FileAttachment file={jobDescriptionFile} onRemove={handleRemoveFile} />
          </div>
        ) : (
          <div
            className={`jd-textarea-dropzone ${dragActive ? 'jd-textarea-dropzone--active' : ''}`}
            onDragOver={(event) => {
              event.preventDefault()
              setDragActive(true)
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
          >
            <textarea
              id="job-description"
              className="textarea-input"
              placeholder="Paste or type the job description here, or drop a .txt file..."
              rows={10}
              value={jobDescriptionText}
              onChange={(event) => onJobDescriptionTextChange(event.target.value)}
            />
            <div className="jd-upload-row">
              <button
                type="button"
                className="link-button"
                onClick={() => fileInputRef.current?.click()}
              >
                Upload .txt file instead
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,text/plain"
                className="hidden-file-input"
                onChange={handleInputChange}
              />
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

export default JobDescriptionForm
