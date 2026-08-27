#!/usr/bin/env python3
"""
Priority 6 diagnostic: traces years_relevant end-to-end using the REAL,
unmodified production code (src/agents/experience_eval.py) against the
real sample resumes/JDs.

job_requirements are hand-built to mirror JobAnalyzerAgent's typical
output for these two JDs (same style already used in
tests/test_batch7_role_relevance_and_soft_skills.py and
tests/test_batch8_title_score_fix.py for jd_01) since no LLM API key is
available in this environment -- title/required_skills/preferred_skills/
min_years_experience/responsibilities are taken directly from the JD text
itself, not invented.
"""
import sys
sys.path.insert(0, ".")

from src.document_parser import parse_document
from src.agents.resume_parser import parse_resume_text
from src.agents.experience_eval import (
    evaluate_experience, job_relevance, job_years, _experience_score_for_years,
    _years_relevant_credit,
)
from src.models import JobRequirements

JD_BACKEND_PYTHON = JobRequirements(
    title="Backend Engineer - Python",
    required_skills=[
        "2+ years of professional experience with Python",
        "Experience building REST APIs (Django REST Framework or FastAPI preferred)",
        "Strong SQL skills and experience with relational databases (PostgreSQL)",
        "Familiarity with Git version control",
        "Understanding of software development best practices",
        "Good communication skills in English",
    ],
    preferred_skills=[
        "Experience with Django or FastAPI frameworks",
        "Familiarity with Docker and containerization",
        "Exposure to message queues (Redis, RabbitMQ, Celery)",
        "Basic AWS/cloud experience (EC2, S3, RDS)",
        "Experience with CI/CD pipelines",
        "Knowledge of microservices architecture",
        "Contributions to open source projects",
    ],
    min_years_experience=2,
    responsibilities=[
        "Design, build, and maintain RESTful APIs using Python",
        "Write clean, well-tested, production-quality code",
        "Work with PostgreSQL databases - schema design, query optimization",
        "Collaborate with frontend team on API contracts",
        "Participate in code reviews and technical discussions",
        "Debug and resolve production issues",
        "Contribute to system design and architecture decisions",
        "Write technical documentation",
    ],
)

JD_FINTECH_SENIOR = JobRequirements(
    title="Senior Backend Engineer - Fintech",
    required_skills=[
        "4+ years building backend systems in Python",
        "2+ years experience in fintech/payments domain",
        "Expert-level Django or FastAPI knowledge",
        "Deep PostgreSQL expertise (replication, partitioning, performance tuning)",
        "Experience designing distributed systems",
        "Strong understanding of security best practices",
        "Experience with high-throughput systems (10,000+ TPS)",
    ],
    preferred_skills=[
        "Previous experience at payment gateways (Razorpay, PayU, etc.)",
        "Knowledge of UPI, IMPS, NEFT protocols",
        "Experience with RBI compliance requirements",
        "Contributions to system design at previous companies",
        "Published technical writing or conference talks",
    ],
    min_years_experience=4,
    responsibilities=[
        "Architect and build payment processing microservices",
        "Design systems for 99.99% uptime and zero data loss",
        "Lead code reviews and establish engineering standards",
        "Mentor junior team members",
        "Work with security team on PCI-DSS compliance",
        "On-call rotation for critical systems",
        "Interface with banking partners on technical integrations",
    ],
)

RESUMES = [
    ("Priya Sharma", "sample_data/resumes/resume_01_priya_sharma.pdf"),
    ("Ananya Patel", "sample_data/resumes/resume_03_ananya_patel.pdf"),
    ("Vikram Singh", "sample_data/resumes/resume_04_vikram_singh.pdf"),
    ("John Doe", "sample_data/resumes/senior_python_dev.txt"),
]

JDS = [
    ("Backend Engineer - Python (2-4y)", JD_BACKEND_PYTHON),
    ("Senior Backend Engineer - Fintech (4-7y)", JD_FINTECH_SENIOR),
]


def hr(t=""):
    print("\n" + "=" * 90)
    if t:
        print(t)
        print("=" * 90)


for cand_name, path in RESUMES:
    parsed = parse_document(path)
    data = parse_resume_text(parsed.text)

    for jd_name, jd in JDS:
        hr(f"{cand_name}  vs  {jd_name}")
        print(f"years_required (min_years_experience) = {jd.min_years_experience}")
        print(f"{'title':<45} {'job_years':>10} {'job_relevance':>14} {'credit(rel)':>13} {'contribution':>13}")
        total = 0.0
        for exp in data.work_experience:
            years = job_years(exp)
            rel = job_relevance(exp, jd)
            credit = _years_relevant_credit(rel)
            contribution = years * credit
            total += contribution
            print(f"{exp.title[:44]:<45} {years:>10.2f} {rel:>14.2f} {credit:>13.3f} {contribution:>13.2f}")
        print(f"{'SUM (years_relevant, rounded)':<45} {'':>10} {'':>14} {'':>13} {round(total,2):>13.2f}")

        ev = evaluate_experience(data, jd)
        print(f"\n-> evaluate_experience(): years_relevant={ev.years_relevant}  "
              f"years_required={ev.years_required}  experience_score={ev.experience_score}  "
              f"role_relevance={ev.role_relevance}")
        print(f"   experience_score formula check: "
              f"_experience_score_for_years({ev.years_relevant}, {ev.years_required}) = "
              f"{_experience_score_for_years(ev.years_relevant, ev.years_required)}")
        print(f"   gaps_identified: {ev.gaps_identified}")
