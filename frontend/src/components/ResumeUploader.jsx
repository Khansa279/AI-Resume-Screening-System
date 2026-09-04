import { useRef, useState } from 'react'
import FileAttachment from './FileAttachment'
import { isAcceptedResumeFile } from '../utils/fileHelpers'
import './ResumeUploader.css'

function ResumeUploader({ files, onFilesAdded, onFileRemoved }) {
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const addFiles = (fileList) => {
    const incoming = Array.from(fileList || [])
    const accepted = []
    const rejected = []

    incoming.forEach((file) => {
      if (isAcceptedResumeFile(file)) {
        accepted.push(file)
      } else {
        rejected.push(file.name)
      }
    })

    if (rejected.length > 0) {
      window.alert(
        `Unsupported file type for: ${rejected.join(', ')}. Only PDF, DOCX, and TXT are supported.`
      )
    }

    if (accepted.length > 0) {
      onFilesAdded(accepted)
    }
  }

  const handleInputChange = (event) => {
    addFiles(event.target.files)
    event.target.value = ''
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setDragActive(false)
    addFiles(event.dataTransfer.files)
  }

  return (
    <section className="resume-uploader">
      <div className="field-label-row">
        <label className="field-label">Resumes</label>
        <span className="field-hint">PDF, DOCX, or TXT &mdash; multiple files allowed</span>
      </div>

      <div
        className={`resume-dropzone ${dragActive ? 'resume-dropzone--active' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            inputRef.current?.click()
          }
        }}
      >
        <span className="resume-dropzone__icon" aria-hidden="true">
          📎
        </span>
        <p className="resume-dropzone__text">
          <strong>Click to upload</strong> or drag and drop resumes here
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.doc,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword,text/plain"
          className="hidden-file-input"
          onChange={handleInputChange}
        />
      </div>

      {files.length > 0 && (
        <div className="resume-attachments">
          {files.map((file, index) => (
            <FileAttachment
              key={`${file.name}-${file.size}-${index}`}
              file={file}
              onRemove={() => onFileRemoved(index)}
            />
          ))}
        </div>
      )}
    </section>
  )
}

export default ResumeUploader
