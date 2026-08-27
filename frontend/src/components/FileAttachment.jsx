import { getFileIcon, formatFileSize } from '../utils/fileHelpers'
import './FileAttachment.css'

/**
 * A single attachment chip -- shows file icon, name, size, and a
 * remove ("x") button. Styled to resemble ChatGPT's file-attachment UI.
 */
function FileAttachment({ file, onRemove }) {
  return (
    <div className="file-attachment">
      <span className="file-attachment__icon" aria-hidden="true">
        {getFileIcon(file.name)}
      </span>
      <div className="file-attachment__info">
        <span className="file-attachment__name" title={file.name}>
          {file.name}
        </span>
        <span className="file-attachment__meta">{formatFileSize(file.size)}</span>
      </div>
      <button
        type="button"
        className="file-attachment__remove"
        aria-label={`Remove ${file.name}`}
        onClick={onRemove}
      >
        ×
      </button>
    </div>
  )
}

export default FileAttachment
