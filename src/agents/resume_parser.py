"""Resume Parser Agent - Extracts structured information from raw resume text.

Normal path is deterministic (regex + section detection). The LLM is only
used as a fallback when the rule-based parse is clearly incomplete.
"""

from __future__ import annotations

import re
from typing import Any

from .base import BaseAgent
from ..models import ResumeData, ContactInfo, Education, WorkExperience
from ..skill_taxonomy import extract_canonical_skills


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}"
)
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-/]+", re.I)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-/]+", re.I)
_GPA_RE = re.compile(r"GPA[:\s]*([0-9]\.\d+)\s*(?:/\s*[0-9]\.\d+)?", re.I)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

_HEADING_RE = re.compile(
    r"^(?:"
    r"professional summary|summary|objective|profile|"
    r"work experience|professional experience|experience|employment|"
    r"internship experience|internships|"
    r"education|academic background|education\s*(?:&|and)\s*training|"
    r"technical skills|skills|core competencies|core skills|"
    r"certifications|certificates|"
    r"projects|personal projects|"
    r"awards|activities|languages|"
    r"career break|career gap|employment gap|sabbatical"
    r")\s*:?\s*$",
    re.I,
)

_SECTION_ALIASES = {
    "professional summary": "summary",
    "summary": "summary",
    "objective": "summary",
    "profile": "summary",
    "work experience": "experience",
    "professional experience": "experience",
    "experience": "experience",
    "employment": "experience",
    "internship experience": "experience",
    "internships": "experience",
    "education": "education",
    "academic background": "education",
    "education & training": "education",
    "education and training": "education",
    "technical skills": "skills",
    "skills": "skills",
    "core competencies": "skills",
    "core skills": "skills",
    "certifications": "certifications",
    "certificates": "certifications",
    "projects": "projects",
    "personal projects": "projects",
    # Recognized so the heading itself (and everything under it) stops
    # leaking into whichever section was previously active -- e.g.
    # "CAREER BREAK" was previously unrecognized entirely, so its text
    # kept accumulating into the still-open "experience" section and
    # occasionally got misparsed by _parse_experience() as a fake job
    # (see _parse_experience's title/company heuristic). There is no
    # ResumeData field for this content, so it's intentionally routed to
    # a section key ("career_break") that nothing downstream reads --
    # dropped, not converted into a WorkExperience or Education entry.
    "career break": "career_break",
    "career gap": "career_break",
    "employment gap": "career_break",
    "sabbatical": "career_break",
}

_DEGREE_RE = re.compile(
    r"(bachelor|master|phd|doctor|b\.?s\.?|m\.?s\.?|bsc|msc|b\.?tech|m\.?tech|"
    r"associate|mba|b\.?e\.?)",
    re.I,
)
_DATE_LINE_RE = re.compile(
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|(?:19|20)\d{2}|present|current|summer)",
    re.I,
)

# Defense-in-depth guard for _parse_experience(): even with the heading
# recognition above, a resume can still use SOME heading variant we
# haven't enumerated (or omit a heading altogether around a short
# paragraph), leaking education/training or career-break text into the
# "experience" bucket that _parse_experience()'s title+company heuristic
# then misreads as a fake job. These two checks catch that content at
# the point a WorkExperience would otherwise be fabricated, without
# touching the title/company/date parsing heuristic itself for anything
# that doesn't match.
#
# Deliberately NOT reusing the bare _DEGREE_RE from education parsing
# here: _DEGREE_RE includes short/ambiguous words ("associate", "b.e.")
# that legitimately appear in real job titles ("Associate Software
# Engineer"). This is a narrower pattern, and -- for "master" -- requires
# "master of"/"master's" rather than the bare word, so a real "Scrum
# Master" title is never affected.
_EDUCATION_ENTRY_TITLE_RE = re.compile(
    r"(bachelor|master\s+of|master'?s\b|ph\.?d\b|b\.?sc\b|m\.?sc\b|"
    r"b\.?tech\b|m\.?tech\b|\bmba\b|diploma|bootcamp)",
    re.I,
)
_INSTITUTION_NAME_RE = re.compile(
    r"(university|college|institute|academy|polytechnic|bootcamp)",
    re.I,
)
_NON_JOB_TITLE_RE = re.compile(
    r"^(?:career break|career gap|employment gap|sabbatical)\b",
    re.I,
)


def _looks_like_education_or_break_entry(title: str, company: str) -> bool:
    """True when a would-be WorkExperience's title/company look like a
    degree/training entry or a career-break note that leaked out of its
    real section, rather than an actual job."""
    if _NON_JOB_TITLE_RE.match(title.strip()):
        return True
    return bool(_EDUCATION_ENTRY_TITLE_RE.search(title) and _INSTITUTION_NAME_RE.search(company))


def _split_sections(text: str) -> dict[str, str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: dict[str, list[str]] = {"_header": []}
    current = "_header"
    for line in lines:
        stripped = line.strip()
        if stripped and _HEADING_RE.match(stripped):
            key = _SECTION_ALIASES.get(re.sub(r"\s+", " ", stripped).lower().rstrip(":"), None)
            if key:
                current = key
                sections.setdefault(current, [])
                continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def _parse_contact(header: str, full_text: str) -> ContactInfo:
    blob = header or full_text[:800]
    email = (_EMAIL_RE.search(full_text) or _EMAIL_RE.search(blob))
    phone = _PHONE_RE.search(blob) or _PHONE_RE.search(full_text)
    linkedin = _LINKEDIN_RE.search(full_text)
    github = _GITHUB_RE.search(full_text)

    name = ""
    for line in blob.split("\n"):
        candidate = line.strip().strip("|").strip()
        if not candidate or "@" in candidate or _HEADING_RE.match(candidate):
            continue
        if _PHONE_RE.search(candidate) and "@" in candidate:
            continue
        words = [w for w in re.split(r"\s+", candidate) if w]
        if 2 <= len(words) <= 5 and not _DEGREE_RE.search(candidate):
            if all(re.match(r"^[A-Za-z][A-Za-z.\-']*$", w) for w in words if w.lower() not in {"|"}):
                name = " ".join(words)
                break
            # Allow "KHANSA ASLAM" / mixed punctuation-light names
            if all(any(c.isalpha() for c in w) for w in words) and len(candidate) < 60:
                if not re.search(r"\d", candidate) and "http" not in candidate.lower():
                    name = candidate
                    break

    location = ""
    loc = re.search(
        r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*),\s*([A-Z]{2}|[A-Z][a-zA-Z]+)\b",
        blob,
    )
    if loc:
        location = loc.group(0)

    return ContactInfo(
        name=name,
        email=email.group(0) if email else "",
        phone=phone.group(0).strip() if phone else "",
        location=location,
        linkedin=linkedin.group(0) if linkedin else "",
        github=github.group(0) if github else "",
    )


def _parse_skills_section(text: str) -> list[str]:
    if not text:
        return []
    items: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip().lstrip("-•*").strip()
        if not line:
            continue
        if ":" in line:
            line = line.split(":", 1)[1]
        for part in re.split(r"[,|/]|•", line):
            skill = part.strip().strip(".")
            skill = re.sub(r"\s*\((?:basic|advanced|intermediate|expert)\)\s*", "", skill, flags=re.I)
            if skill and 1 < len(skill) < 60:
                items.append(skill)
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _parse_education(text: str) -> list[Education]:
    if not text:
        return []
    blocks = re.split(r"\n\s*\n", text)
    education: list[Education] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        joined = " ".join(lines)
        if not _DEGREE_RE.search(joined) and "university" not in joined.lower() and "college" not in joined.lower():
            continue
        degree = ""
        field = ""
        first = lines[0]
        if " in " in first.lower():
            left, right = re.split(r"\s+in\s+", first, maxsplit=1, flags=re.I)
            degree, field = left.strip(), right.strip()
        else:
            degree = first
        institution = ""
        for line in lines[1:]:
            if "|" in line:
                institution = line.split("|")[0].strip()
                break
            if "university" in line.lower() or "college" in line.lower() or "institute" in line.lower():
                institution = line.split("|")[0].strip()
                break
        if not institution and len(lines) > 1:
            institution = lines[1].split("|")[0].strip()
        year_m = _YEAR_RE.search(joined)
        gpa_m = _GPA_RE.search(joined)
        education.append(Education(
            degree=degree,
            field=field,
            institution=institution,
            graduation_year=year_m.group(1) if year_m else "",
            gpa=gpa_m.group(0) if gpa_m else "",
        ))
    return education


def _technologies_for_job(title: str, bullets: list[str]) -> list[str]:
    """Deterministically recognize technologies/tools/technical concepts
    mentioned in one job's title + responsibility bullets, via the shared
    skill taxonomy.

    Previously WorkExperience.technologies was always hardcoded to [],
    which meant ExperienceEvaluatorAgent.job_relevance()'s skill_hit
    component (40% of role_relevance) never had anything to compare
    against and silently contributed ~0 for every candidate. Soft skills
    are excluded here (Leadership, Teamwork, ...) since a "technologies"
    list should stay technologies -- those still get captured separately
    by SkillExtractorAgent from the same bullet text.
    """
    text = " ".join([title or "", " ".join(bullets)])
    hits = extract_canonical_skills(text)
    return [name for name, category in hits if category != "soft_skill"]


def _parse_experience(text: str) -> list[WorkExperience]:
    if not text:
        return []
    lines = [ln.rstrip() for ln in text.split("\n")]
    jobs: list[WorkExperience] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith(("-", "•", "*")):
            i += 1
            continue
        # Title line: next lines may be company | location | dates
        title = line.lstrip("-•*").strip()
        company = ""
        duration = ""
        start_date = ""
        end_date = ""
        look = lines[i + 1].strip() if i + 1 < len(lines) else ""
        consumed = 0
        if look and not look.startswith(("-", "•", "*")):
            consumed = 1
            parts = [p.strip() for p in re.split(r"\s*\|\s*", look)]
            if parts:
                company = parts[0]
            date_blob = look
            if len(parts) >= 2:
                date_blob = parts[-1]
            duration = date_blob
            left, right = (re.split(r"\s*(?:-|–|—|to)\s*", date_blob, maxsplit=1) + [""])[:2]
            start_date, end_date = left.strip(), right.strip()
        bullets: list[str] = []
        j = i + 1 + consumed
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt:
                # blank: maybe next job
                j += 1
                if j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(("-", "•", "*")):
                    break
                continue
            if nxt.startswith(("-", "•", "*")):
                bullets.append(nxt.lstrip("-•* ").strip())
                j += 1
                continue
            break
        if title and (company or bullets or _DATE_LINE_RE.search(duration)) and not _looks_like_education_or_break_entry(title, company):
            jobs.append(WorkExperience(
                title=title,
                company=company or "Unknown",
                duration=duration,
                start_date=start_date,
                end_date=end_date,
                responsibilities=bullets,
                technologies=_technologies_for_job(title, bullets),
            ))
            i = j
            continue
        i += 1
    return jobs


def _parse_list_section(text: str) -> list[str]:
    if not text:
        return []
    items: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if current:
                items.append(" ".join(current).strip())
                current = []
            continue
        if stripped.startswith(("-", "•", "*")):
            if current:
                items.append(" ".join(current).strip())
            current = [stripped.lstrip("-•* ").strip()]
        else:
            if current:
                current.append(stripped)
            else:
                current = [stripped]
    if current:
        items.append(" ".join(current).strip())
    return [i for i in items if i]


def parse_resume_text(raw_text: str) -> ResumeData:
    """Deterministic structured parse of resume text."""
    sections = _split_sections(raw_text)
    header = sections.get("_header", raw_text[:600])
    contact = _parse_contact(header, raw_text)
    summary = sections.get("summary", "")
    if summary:
        summary = " ".join(ln.strip() for ln in summary.split("\n") if ln.strip())[:800]

    education = _parse_education(sections.get("education", ""))
    work = _parse_experience(sections.get("experience", ""))
    skills = _parse_skills_section(sections.get("skills", ""))
    certifications = _parse_list_section(sections.get("certifications", ""))
    projects = _parse_list_section(sections.get("projects", ""))

    notes: list[str] = []
    if not contact.email:
        notes.append("Email not found by rule-based parser")
    if not work:
        notes.append("No work/internship section parsed")
    confidence = 0.9
    if notes:
        confidence = 0.7
    if not contact.name and not contact.email:
        confidence = 0.4

    return ResumeData(
        contact=contact,
        summary=summary,
        education=education,
        work_experience=work,
        skills_section=skills,
        certifications=certifications,
        projects=projects,
        raw_text=raw_text,
        parsing_confidence=confidence,
        parsing_notes=notes or ["Parsed with deterministic rules"],
    )


def _parse_is_sufficient(data: ResumeData) -> bool:
    has_identity = bool(data.contact.email or (data.contact.name and " " in data.contact.name.strip()))
    has_substance = bool(data.work_experience or data.education or data.skills_section)
    return has_identity and has_substance


class ResumeParserAgent(BaseAgent):
    """
    Agent responsible for parsing raw resume text into structured data.
    
    This agent takes messy, unstructured resume text and extracts:
    - Contact information
    - Education history
    - Work experience
    - Skills section
    - Certifications
    - Projects
    
    It handles various resume formats and messy data gracefully.
    """
    
    name = "ResumeParserAgent"
    description = "Parse raw resume text into structured sections (contact, education, experience, skills)"
    temperature = 0.0
    
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Parse the resume and extract structured information.
        
        Args:
            state: Current workflow state containing resume_raw_text
            
        Returns:
            State updates with parsed resume_data
        """
        raw_text = state.get("resume_raw_text", "")
        
        if not raw_text:
            return {
                "resume_data": ResumeData(
                    parsing_confidence=0.0,
                    parsing_notes=["No resume text provided"]
                ),
                "errors": ["ResumeParserAgent: No resume text to parse"],
                "agent_confidences": {self.name: 0.0}
            }

        resume_data = parse_resume_text(raw_text)
        if not _parse_is_sufficient(resume_data):
            prompt = self._build_parsing_prompt(raw_text)
            response = await self._call_llm_for_json_async(prompt)
            llm_data = self._parse_response(response, raw_text)
            if llm_data.parsing_confidence >= resume_data.parsing_confidence:
                if "Failed to parse" not in " ".join(llm_data.parsing_notes):
                    llm_data.parsing_notes = list(llm_data.parsing_notes) + [
                        "Used LLM fallback after incomplete rule-based parse"
                    ]
                    resume_data = llm_data

        return {
            "resume_data": resume_data,
            "agent_confidences": {self.name: resume_data.parsing_confidence}
        }
    
    def _build_parsing_prompt(self, raw_text: str) -> str:
        """Build the prompt for resume parsing."""
        return f"""{self._build_system_prompt()}

TASK: Parse the following resume text into structured JSON format.

RESUME TEXT:
---
{raw_text[:8000]}  
---

Extract the following information and return as JSON:
{{
    "contact": {{
        "name": "Full name of the candidate",
        "email": "Email address",
        "phone": "Phone number",
        "location": "City, State/Country",
        "linkedin": "LinkedIn URL if present",
        "github": "GitHub URL if present"
    }},
    "summary": "Professional summary or objective if present",
    "education": [
        {{
            "degree": "Degree type (e.g., BS, MS, PhD)",
            "field": "Field of study",
            "institution": "School/University name",
            "graduation_year": "Year of graduation",
            "gpa": "GPA if mentioned"
        }}
    ],
    "work_experience": [
        {{
            "title": "Job title",
            "company": "Company name",
            "duration": "How long in this role",
            "start_date": "Start date",
            "end_date": "End date or 'Present'",
            "responsibilities": ["List of key responsibilities"],
            "technologies": ["Technologies/tools used in this role"]
        }}
    ],
    "skills_section": ["List of explicitly mentioned skills"],
    "certifications": ["List of certifications"],
    "projects": ["Project Name -- one-sentence description of what it does and the key technologies used"],
    "parsing_confidence": 0.0 to 1.0 (how confident are you in this extraction),
    "parsing_notes": ["Any issues or uncertainties in parsing"]
}}

IMPORTANT:
- Extract what you can find, leave empty strings/arrays for missing info
- List work experience in reverse chronological order
- Include ALL skills mentioned anywhere in the resume
- For projects, ALWAYS use the exact format "Name -- one-sentence description",
  every single time, with no exceptions. If the resume only lists a bare project
  name with no description, write a short one-sentence description yourself based
  on the name/context -- never return the bare name alone.
- Note any formatting issues or missing sections in parsing_notes
- Set parsing_confidence based on how complete and clear the resume was

Respond with ONLY valid JSON, no additional text."""
    
    def _parse_response(self, response: str, raw_text: str) -> ResumeData:
        """Parse the LLM response into a ResumeData object."""
        data = self._extract_json_from_response(response)
        
        if not data:
            # If JSON parsing fails, return a minimal result
            return ResumeData(
                raw_text=raw_text,
                parsing_confidence=0.3,
                parsing_notes=["Failed to parse LLM response as JSON"]
            )
        
        try:
            # Parse contact info
            contact_data = data.get("contact", {})
            contact = ContactInfo(
                name=contact_data.get("name", ""),
                email=contact_data.get("email", ""),
                phone=contact_data.get("phone", ""),
                location=contact_data.get("location", ""),
                linkedin=contact_data.get("linkedin", ""),
                github=contact_data.get("github", "")
            )
            
            # Parse education
            education = []
            for edu in data.get("education", []):
                education.append(Education(
                    degree=edu.get("degree", ""),
                    field=edu.get("field", ""),
                    institution=edu.get("institution", ""),
                    graduation_year=edu.get("graduation_year", ""),
                    gpa=edu.get("gpa", "")
                ))
            
            # Parse work experience
            work_experience = []
            for exp in data.get("work_experience", []):
                work_experience.append(WorkExperience(
                    title=exp.get("title", "Unknown"),
                    company=exp.get("company", "Unknown"),
                    duration=exp.get("duration", ""),
                    start_date=exp.get("start_date", ""),
                    end_date=exp.get("end_date", ""),
                    responsibilities=exp.get("responsibilities", []),
                    technologies=exp.get("technologies", [])
                ))
            
            return ResumeData(
                contact=contact,
                summary=data.get("summary", ""),
                education=education,
                work_experience=work_experience,
                skills_section=data.get("skills_section", []),
                certifications=data.get("certifications", []),
                projects=data.get("projects", []),
                raw_text=raw_text,
                parsing_confidence=float(data.get("parsing_confidence", 0.7)),
                parsing_notes=data.get("parsing_notes", [])
            )
        except Exception as e:
            return ResumeData(
                raw_text=raw_text,
                parsing_confidence=0.3,
                parsing_notes=[f"Error constructing ResumeData: {str(e)}"]
            )
