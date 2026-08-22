"""
SQLAlchemy ORM models for the recruitment platform.

This is the persistent counterpart to the transient Pydantic models in
src/models.py. The two are intentionally similar in shape — src/models.py
is what agents pass to each other in memory during one screening run;
this file is what survives after the process exits.

Hierarchy:

    Organization
      -> Department
        -> Position
          -> JobDescription (versioned)
            -> JobRequirements (1:1 with a JobDescription version)
              -> JobRequirementSkill (many)
              -> JobRequirementItem (many)

    Candidate
      -> Resume (submitted for a specific Position)
        -> ResumeEducation (many)
        -> ResumeWorkExperience (many)
        -> CandidateSkill (many)

    Screening (ties one Resume to one JobDescription version)
      -> SkillMatchResult (1:1)
        -> SkillMatch (many)
      -> ExperienceEvaluation (1:1)
      -> ScreeningResult (1:1) -- the final score/recommendation
      -> Explanation (1:1) -- structured "5 Ws" evidence

    Ranking (one ranking run for a JobDescription version)
      -> RankingEntry (many, one per Screening included in that ranking)
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow() -> datetime:
    """Single source of truth for 'now' so every default timestamp is UTC."""
    return datetime.now(timezone.utc)


# ============================================================================
# Organization hierarchy
# ============================================================================

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    departments = relationship(
        "Department", back_populates="organization", cascade="all, delete-orphan"
    )


class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    organization = relationship("Organization", back_populates="departments")
    positions = relationship(
        "Position", back_populates="department", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("organization_id", "name", name="uq_dept_org_name"),)


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=False)
    title = Column(String, nullable=False)
    status = Column(String, default="open", nullable=False)  # open | closed | draft
    created_at = Column(DateTime, default=utcnow, nullable=False)

    department = relationship("Department", back_populates="positions")
    job_descriptions = relationship(
        "JobDescription", back_populates="position", cascade="all, delete-orphan"
    )
    resumes = relationship("Resume", back_populates="position")


# ============================================================================
# Job description + requirements
# ============================================================================

class JobDescription(Base):
    """A versioned JD. Editing a JD creates a new row rather than mutating
    the old one, so past screenings stay linked to the exact JD text they
    were evaluated against."""

    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=False)
    raw_text = Column(Text, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    is_current = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    position = relationship("Position", back_populates="job_descriptions")
    requirements = relationship(
        "JobRequirements",
        back_populates="job_description",
        uselist=False,
        cascade="all, delete-orphan",
    )
    screenings = relationship("Screening", back_populates="job_description")
    rankings = relationship("Ranking", back_populates="job_description")

    __table_args__ = (
        UniqueConstraint("position_id", "version", name="uq_jd_position_version"),
    )


class JobRequirements(Base):
    """Structured extraction of one JobDescription version.
    Mirrors src/models.py::JobRequirements (top-level fields only;
    skill lists and granular requirements are child tables below)."""

    __tablename__ = "job_requirements"

    id = Column(Integer, primary_key=True)
    job_description_id = Column(
        Integer, ForeignKey("job_descriptions.id"), nullable=False, unique=True
    )
    title = Column(String, default="")
    summary = Column(Text, default="")
    min_years_experience = Column(Integer, default=0)
    education_requirements = Column(JSON, default=list)  # list[str]
    certifications_required = Column(JSON, default=list)  # list[str]
    responsibilities = Column(JSON, default=list)  # list[str]
    parsing_confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    job_description = relationship("JobDescription", back_populates="requirements")
    skills = relationship(
        "JobRequirementSkill", back_populates="job_requirements", cascade="all, delete-orphan"
    )
    items = relationship(
        "JobRequirementItem", back_populates="job_requirements", cascade="all, delete-orphan"
    )


class JobRequirementSkill(Base):
    """One required or preferred skill for a JD. Denormalized (free-text
    skill_name) by design for now -- see architecture notes."""

    __tablename__ = "job_requirement_skills"

    id = Column(Integer, primary_key=True)
    job_requirements_id = Column(Integer, ForeignKey("job_requirements.id"), nullable=False)
    skill_name = Column(String, nullable=False)
    priority = Column(String, nullable=False)  # required | preferred
    category = Column(String, default="other")

    job_requirements = relationship("JobRequirements", back_populates="skills")


class JobRequirementItem(Base):
    """Mirrors src/models.py::Requirement -- a single granular requirement
    line item (skill / experience / education / certification / other)."""

    __tablename__ = "job_requirement_items"

    id = Column(Integer, primary_key=True)
    job_requirements_id = Column(Integer, ForeignKey("job_requirements.id"), nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String, default="other")
    priority = Column(String, default="required")  # required | preferred | nice_to_have
    years_needed = Column(Integer, nullable=True)

    job_requirements = relationship("JobRequirements", back_populates="items")


# ============================================================================
# Candidate + resume
# ============================================================================

class Candidate(Base):
    """A person. Kept separate from Resume so the same person applying to
    multiple positions (or re-applying) doesn't create disconnected records."""

    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True)
    name = Column(String, default="")
    email = Column(String, nullable=True, index=True)
    phone = Column(String, default="")
    location = Column(String, default="")
    linkedin = Column(String, default="")
    github = Column(String, default="")
    created_at = Column(DateTime, default=utcnow, nullable=False)

    resumes = relationship("Resume", back_populates="candidate", cascade="all, delete-orphan")


class Resume(Base):
    """One resume submission, tied to a specific candidate and the position
    they applied for. file_path points at the stored file on disk;
    raw_text is the extracted text (kept for re-processing without
    re-parsing the original file)."""

    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=False)
    file_path = Column(String, nullable=True)
    file_type = Column(String, default="unknown")  # pdf | docx | txt | unknown
    raw_text = Column(Text, default="")
    # SHA-256 of whitespace-normalized extracted text. Indexed (not unique)
    # so existing development duplicates can be backfilled; application
    # logic reuses the earliest matching row instead of inserting another.
    content_hash = Column(String, nullable=True, index=True)
    summary = Column(Text, default="")
    skills_section = Column(JSON, default=list)  # list[str], as explicitly listed on resume
    certifications = Column(JSON, default=list)  # list[str]
    projects = Column(JSON, default=list)  # list[str]
    parsing_confidence = Column(Float, default=0.0)
    parsing_notes = Column(JSON, default=list)  # list[str]
    uploaded_at = Column(DateTime, default=utcnow, nullable=False)

    candidate = relationship("Candidate", back_populates="resumes")
    position = relationship("Position", back_populates="resumes")
    education = relationship(
        "ResumeEducation", back_populates="resume", cascade="all, delete-orphan"
    )
    work_experience = relationship(
        "ResumeWorkExperience", back_populates="resume", cascade="all, delete-orphan"
    )
    skills = relationship(
        "CandidateSkill", back_populates="resume", cascade="all, delete-orphan"
    )
    screenings = relationship("Screening", back_populates="resume")


class ResumeEducation(Base):
    __tablename__ = "resume_education"

    id = Column(Integer, primary_key=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=False)
    degree = Column(String, default="")
    field = Column(String, default="")
    institution = Column(String, default="")
    graduation_year = Column(String, default="")
    gpa = Column(String, default="")

    resume = relationship("Resume", back_populates="education")


class ResumeWorkExperience(Base):
    __tablename__ = "resume_work_experience"

    id = Column(Integer, primary_key=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=False)
    title = Column(String, nullable=False)
    company = Column(String, nullable=False)
    duration = Column(String, default="")
    start_date = Column(String, default="")
    end_date = Column(String, default="")
    responsibilities = Column(JSON, default=list)  # list[str]
    technologies = Column(JSON, default=list)  # list[str]

    resume = relationship("Resume", back_populates="work_experience")


class CandidateSkill(Base):
    """Mirrors src/models.py::Skill, scoped to one resume submission."""

    __tablename__ = "candidate_skills"

    id = Column(Integer, primary_key=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=False)
    name = Column(String, nullable=False)
    category = Column(String, default="other")
    proficiency = Column(String, default="intermediate")
    confidence = Column(Float, default=0.8)
    source = Column(String, default="explicit")  # explicit | inferred

    resume = relationship("Resume", back_populates="skills")


# ============================================================================
# Screening / evaluation
# ============================================================================

class Screening(Base):
    """One evaluation event: this Resume against this JobDescription
    version. Everything downstream (scores, ranking entries, explanation)
    hangs off this row's id."""

    __tablename__ = "screenings"

    id = Column(Integer, primary_key=True)
    resume_id = Column(Integer, ForeignKey("resumes.id"), nullable=False)
    job_description_id = Column(Integer, ForeignKey("job_descriptions.id"), nullable=False)
    status = Column(String, default="pending")  # pending | running | completed | failed
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    resume = relationship("Resume", back_populates="screenings")
    job_description = relationship("JobDescription", back_populates="screenings")

    skill_match_result = relationship(
        "SkillMatchResult", back_populates="screening", uselist=False,
        cascade="all, delete-orphan",
    )
    experience_evaluation = relationship(
        "ExperienceEvaluation", back_populates="screening", uselist=False,
        cascade="all, delete-orphan",
    )
    result = relationship(
        "ScreeningResult", back_populates="screening", uselist=False,
        cascade="all, delete-orphan",
    )
    explanation = relationship(
        "Explanation", back_populates="screening", uselist=False,
        cascade="all, delete-orphan",
    )
    ranking_entries = relationship("RankingEntry", back_populates="screening")

    __table_args__ = (
        UniqueConstraint("resume_id", "job_description_id", name="uq_screening_resume_jd"),
    )


class SkillMatchResult(Base):
    """Mirrors src/models.py::SkillsMatchResult."""

    __tablename__ = "skill_match_results"

    id = Column(Integer, primary_key=True)
    screening_id = Column(Integer, ForeignKey("screenings.id"), nullable=False, unique=True)
    required_skills_met = Column(Integer, default=0)
    required_skills_total = Column(Integer, default=0)
    preferred_skills_met = Column(Integer, default=0)
    preferred_skills_total = Column(Integer, default=0)
    overall_score = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    reasoning = Column(Text, default="")

    screening = relationship("Screening", back_populates="skill_match_result")
    matches = relationship(
        "SkillMatch", back_populates="skill_match_result", cascade="all, delete-orphan"
    )


class SkillMatch(Base):
    """Mirrors src/models.py::SkillMatch -- one requirement's match detail."""

    __tablename__ = "skill_matches"

    id = Column(Integer, primary_key=True)
    skill_match_result_id = Column(
        Integer, ForeignKey("skill_match_results.id"), nullable=False
    )
    requirement = Column(String, default="")
    matched = Column(Boolean, default=False)
    matched_skill = Column(String, default="")
    match_quality = Column(String, default="none")  # exact | semantic | partial | none
    confidence = Column(Float, default=0.0)
    notes = Column(Text, default="")

    skill_match_result = relationship("SkillMatchResult", back_populates="matches")


class ExperienceEvaluation(Base):
    """Mirrors src/models.py::ExperienceEvaluation."""

    __tablename__ = "experience_evaluations"

    id = Column(Integer, primary_key=True)
    screening_id = Column(Integer, ForeignKey("screenings.id"), nullable=False, unique=True)
    years_relevant = Column(Float, default=0.0)
    years_required = Column(Integer, default=0)
    experience_score = Column(Float, default=0.0)
    role_relevance = Column(Float, default=0.0)
    career_progression = Column(Text, default="")
    gaps_identified = Column(JSON, default=list)  # list[str]
    strengths = Column(JSON, default=list)  # list[str]
    confidence = Column(Float, default=0.0)
    reasoning = Column(Text, default="")

    screening = relationship("Screening", back_populates="experience_evaluation")


class ScreeningResult(Base):
    """Mirrors src/models.py::ScreeningOutput -- the final deliverable for
    one candidate. This is the row the ranking step reads scores from."""

    __tablename__ = "screening_results"

    id = Column(Integer, primary_key=True)
    screening_id = Column(Integer, ForeignKey("screenings.id"), nullable=False, unique=True)
    match_score = Column(Float, nullable=False)
    recommendation = Column(String, nullable=False)
    requires_human = Column(Boolean, default=False)
    confidence = Column(Float, nullable=False)
    reasoning_summary = Column(Text, default="")
    flags = Column(JSON, default=list)  # list[str]
    created_at = Column(DateTime, default=utcnow, nullable=False)

    screening = relationship("Screening", back_populates="result")


# ============================================================================
# Ranking
# ============================================================================

class Ranking(Base):
    """One ranking run over all screenings for a given JobDescription
    version. Kept separate from individual screenings so re-running the
    ranking (e.g. after new candidates are added) doesn't lose history."""

    __tablename__ = "rankings"

    id = Column(Integer, primary_key=True)
    job_description_id = Column(Integer, ForeignKey("job_descriptions.id"), nullable=False)
    method = Column(String, default="match_score_desc")
    generated_at = Column(DateTime, default=utcnow, nullable=False)

    job_description = relationship("JobDescription", back_populates="rankings")
    entries = relationship(
        "RankingEntry", back_populates="ranking", cascade="all, delete-orphan"
    )


class RankingEntry(Base):
    __tablename__ = "ranking_entries"

    id = Column(Integer, primary_key=True)
    ranking_id = Column(Integer, ForeignKey("rankings.id"), nullable=False)
    screening_id = Column(Integer, ForeignKey("screenings.id"), nullable=False)
    rank_position = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)

    ranking = relationship("Ranking", back_populates="entries")
    screening = relationship("Screening", back_populates="ranking_entries")

    __table_args__ = (
        UniqueConstraint("ranking_id", "screening_id", name="uq_ranking_entry"),
    )


# ============================================================================
# Explanation ("5 Ws")
# ============================================================================

class Explanation(Base):
    """Structured, evidence-linked explanation for one screening, distinct
    from the free-text reasoning_summary on ScreeningResult. This is what
    the recruiter-facing 'why was this candidate ranked here' view reads."""

    __tablename__ = "explanations"

    id = Column(Integer, primary_key=True)
    screening_id = Column(Integer, ForeignKey("screenings.id"), nullable=False, unique=True)
    why_summary = Column(Text, default="")           # WHY they scored as they did
    matching_skills = Column(JSON, default=list)      # WHAT contributed (skills)
    evidence_snippets = Column(JSON, default=list)     # WHERE it appears (resume quotes/sections)
    relevant_experience = Column(Text, default="")     # WHEN/how much relevant experience
    skill_gaps = Column(JSON, default=list)            # what's missing
    confidence_label = Column(String, default="")       # e.g. "High" / "Medium" / "Low"
    generated_at = Column(DateTime, default=utcnow, nullable=False)

    screening = relationship("Screening", back_populates="explanation")
