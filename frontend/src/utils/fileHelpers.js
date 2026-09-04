const EXTENSION_ICON_MAP = {
  pdf: '📄',
  docx: '📝',
  doc: '📝',
  txt: '📃',
}

export function getFileExtension(fileName = '') {
  const parts = fileName.split('.')
  return parts.length > 1 ? parts.pop().toLowerCase() : ''
}

export function getFileIcon(fileName) {
  const ext = getFileExtension(fileName)
  return EXTENSION_ICON_MAP[ext] || '📁'
}

export function formatFileSize(bytes) {
  if (!bytes && bytes !== 0) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export const ACCEPTED_RESUME_EXTENSIONS = ['pdf', 'docx', 'doc', 'txt']
export const ACCEPTED_RESUME_MIME_TYPES = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/msword',
  'text/plain',
]

export function isAcceptedResumeFile(file) {
  const ext = getFileExtension(file.name)
  return ACCEPTED_RESUME_EXTENSIONS.includes(ext)
}

export function isAcceptedJobDescriptionFile(file) {
  return getFileExtension(file.name) === 'txt'
}
