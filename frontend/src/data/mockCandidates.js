/**
 * Placeholder ranking data for the results-preview UI. Nothing here is
 * wired to the API yet -- see src/services/api.js for the real
 * screening/results endpoints this will eventually be replaced by.
 */
export const mockCandidates = [
  {
    id: 'c1',
    rank: 1,
    name: 'Jordan Ade',
    email: 'jordan.ade@example.com',
    phone: '+1 (415) 555-0148',
    resumeFile: 'jordan_ade_resume.pdf',
    matchScore: 0.87,
    recommendation: 'Proceed to interview',
    explanation:
      'Five years of backend experience with the exact stack this role calls for, plus a track record of owning services end to end.',
    matchedSkills: ['Python', 'Django', 'PostgreSQL', 'REST APIs', 'Docker'],
    skillGaps: ['Kubernetes'],
    breakdown: { skillMatch: 0.91, experience: 0.83, roleRelevance: 0.88 },
  },
  {
    id: 'c2',
    rank: 2,
    name: 'Priya Nandakumar',
    email: 'priya.n@example.com',
    phone: '+1 (206) 555-0173',
    resumeFile: 'priya_nandakumar_resume.pdf',
    matchScore: 0.74,
    recommendation: 'Proceed to phone screening',
    explanation:
      'Strong applied ML background and clean Python fundamentals; less direct exposure to production API design than the role expects.',
    matchedSkills: ['Python', 'Machine Learning', 'SQL', 'Git'],
    skillGaps: ['REST APIs', 'Docker'],
    breakdown: { skillMatch: 0.7, experience: 0.79, roleRelevance: 0.72 },
  },
  {
    id: 'c3',
    rank: 3,
    name: 'Marcus Webb',
    email: 'marcus.webb@example.com',
    phone: '+1 (312) 555-0119',
    resumeFile: 'marcus_webb_resume.docx',
    matchScore: 0.52,
    recommendation: 'Needs manual review',
    explanation:
      'Solid generalist engineering background, but limited evidence of the specific frameworks and years of experience this role requires.',
    matchedSkills: ['Python', 'Git'],
    skillGaps: ['Django', 'PostgreSQL', 'REST APIs'],
    breakdown: { skillMatch: 0.48, experience: 0.55, roleRelevance: 0.51 },
  },
]

export const mockJobTitle = 'Backend Engineer — Python'
