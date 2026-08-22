"""
Deterministic skill taxonomy shared by SkillExtractorAgent and
SkillsMatcherAgent.

Why this file exists
---------------------
Skill extraction and skill matching used to each make their own LLM call.
That was the single biggest source of non-determinism in the pipeline
(the same resume text could produce a different skill list -- and
therefore a different match score -- on every run) and it was also two
of the most token-hungry calls per candidate.

This module gives both agents ONE canonical, hand-curated mapping of
"anything a resume/JD might call a skill" -> "the canonical skill name +
category". Both agents import from here so there is a single place to
extend the taxonomy instead of duplicating alias lists.

This is intentionally a plain Python dict, not an embeddings/ML
approach -- per the project constraints we are not adding new
dependencies, and a curated alias table is sufficient to cover the
skills that show up in software-engineering resumes/JDs.
"""

from __future__ import annotations

import re

# ============================================================================
# Canonical taxonomy: canonical_name -> (category, [aliases...])
# Aliases are matched case-insensitively, and are compared against
# word-boundary-normalized text (see `normalize_skill_name`).
# Add new skills/aliases here -- both extraction and matching pick them
# up automatically.
# ============================================================================

TAXONOMY: dict[str, tuple[str, list[str]]] = {
    # ---- Languages ---------------------------------------------------
    "Python": ("language", ["python", "python3", "py"]),
    "JavaScript": ("language", ["javascript", "js", "es6", "ecmascript"]),
    "TypeScript": ("language", ["typescript", "ts"]),
    "Java": ("language", ["java"]),
    "Go": ("language", ["golang", "go lang", "go"]),
    "SQL": ("language", ["sql", "structured query language"]),
    "C++": ("language", ["c++", "cpp"]),
    "C#": ("language", ["c#", "csharp", "c sharp"]),
    "C": ("language", ["c programming", " c "]),
    "Ruby": ("language", ["ruby"]),
    "PHP": ("language", ["php"]),
    "Swift": ("language", ["swift"]),
    "Kotlin": ("language", ["kotlin"]),
    "HTML": ("language", ["html", "html5"]),
    "CSS": ("language", ["css", "css3"]),
    "R": ("language", ["r language", "r programming"]),
    "Scala": ("language", ["scala"]),
    "Rust": ("language", ["rust"]),
    "Bash": ("language", ["bash", "shell scripting", "shell script"]),

    # ---- Frameworks ----------------------------------------------------
    "Django": ("framework", ["django", "django rest framework", "drf"]),
    "Flask": ("framework", ["flask"]),
    "FastAPI": ("framework", ["fastapi", "fast api"]),
    "React": ("framework", ["react", "reactjs", "react.js"]),
    "Angular": ("framework", ["angular", "angularjs"]),
    "Vue.js": ("framework", ["vue", "vuejs", "vue.js"]),
    "Node.js": ("framework", ["node", "nodejs", "node.js"]),
    "Express.js": ("framework", ["express", "expressjs", "express.js"]),
    "Spring": ("framework", ["spring", "spring boot", "springboot"]),
    ".NET": ("framework", [".net", "dotnet", "asp.net"]),
    "LangGraph": ("framework", ["langgraph"]),
    "LangChain": ("framework", ["langchain"]),
    "TensorFlow": ("framework", ["tensorflow", "tf"]),
    "PyTorch": ("framework", ["pytorch", "torch"]),
    "Scikit-learn": ("framework", ["scikit-learn", "sklearn", "scikit learn"]),
    "Pandas": ("framework", ["pandas"]),
    "NumPy": ("framework", ["numpy"]),

    # ---- Tools / Platforms ----------------------------------------------
    "Git": ("tool", ["git", "version control"]),
    "Docker": ("tool", ["docker", "containerization", "containers"]),
    "Kubernetes": ("tool", ["kubernetes", "k8s"]),
    "AWS": ("tool", ["aws", "amazon web services", "ec2", "s3", "lambda", "rds", "sqs"]),
    "GCP": ("tool", ["gcp", "google cloud", "google cloud platform"]),
    "Azure": ("tool", ["azure", "microsoft azure"]),
    "PostgreSQL": ("tool", ["postgresql", "postgres", "psql"]),
    "MySQL": ("tool", ["mysql"]),
    "MongoDB": ("tool", ["mongodb", "mongo"]),
    "Redis": ("tool", ["redis", "caching"]),
    "Kafka": ("tool", ["kafka", "apache kafka"]),
    "RabbitMQ": ("tool", ["rabbitmq", "message queue", "message queues"]),
    "GitHub Actions": ("tool", ["github actions"]),
    "CI/CD": ("tool", ["ci/cd", "continuous integration", "continuous deployment"]),
    "Jenkins": ("tool", ["jenkins"]),
    "Airflow": ("tool", ["airflow", "apache airflow"]),
    "Spark": ("tool", ["spark", "apache spark", "pyspark"]),
    "Heroku": ("tool", ["heroku"]),
    "VS Code": ("tool", ["vs code", "vscode", "visual studio code"]),
    "Jira": ("tool", ["jira"]),
    "Snowflake": ("tool", ["snowflake"]),
    "Databricks": ("tool", ["databricks"]),
    "dbt": ("tool", ["dbt", "data build tool"]),

    # ---- Concepts / "other" technical ------------------------------------
    "REST APIs": ("technical", ["rest api", "rest apis", "restful api", "restful apis", "rest"]),
    "GraphQL": ("technical", ["graphql"]),
    "Microservices": ("technical", ["microservice", "microservices"]),
    "Machine Learning": ("technical", ["machine learning", "ml"]),
    "Deep Learning": ("technical", ["deep learning", "dl"]),
    "Artificial Intelligence": ("technical", ["artificial intelligence", "ai"]),
    "Data Structures": ("technical", ["data structures"]),
    "Algorithms": ("technical", ["algorithms"]),
    "Database Design": ("technical", ["database design", "data modeling", "schema design"]),
    "Data Warehousing": ("technical", ["data warehousing", "data warehouse"]),
    "Agile/Scrum": ("technical", ["agile", "scrum", "agile methodology", "agile/scrum"]),
    "System Design": ("technical", ["system design", "distributed systems"]),
    "MLOps": ("technical", ["mlops"]),
    "ETL": ("technical", ["etl", "data pipelines", "data pipeline"]),

    # ---- Soft skills ------------------------------------------------------
    "Team Leadership": ("soft_skill", ["team leadership", "led team", "leadership", "mentoring", "mentored"]),
    "Communication": ("soft_skill", ["communication skills", "communication"]),
    "Problem Solving": ("soft_skill", ["problem-solving", "problem solving"]),
    "Collaboration": ("soft_skill", ["collaboration", "cross-functional", "teamwork", "team player"]),
    "Code Review": ("soft_skill", ["code review", "code reviews"]),
}

# Build a fast lookup: normalized alias text -> (canonical_name, category)
_ALIAS_LOOKUP: dict[str, tuple[str, str]] = {}
for _canonical, (_category, _aliases) in TAXONOMY.items():
    _ALIAS_LOOKUP[_canonical.lower()] = (_canonical, _category)
    for _alias in _aliases:
        _ALIAS_LOOKUP[_alias.strip().lower()] = (_canonical, _category)

# Sort aliases longest-first so multi-word aliases (e.g. "machine learning")
# are matched before shorter substrings (e.g. "learning") could interfere.
_SORTED_ALIASES = sorted(_ALIAS_LOOKUP.keys(), key=len, reverse=True)


def normalize_skill_name(raw: str) -> tuple[str, str] | None:
    """
    Look up a raw skill string (e.g. "JS", "react.js", "Postgres") and
    return (canonical_name, category) if it's a known skill, else None.

    This is an exact/alias lookup (not a scan) -- use `find_skills_in_text`
    to scan free-form text such as resume responsibilities.
    """
    key = raw.strip().lower()
    key = re.sub(r"\s+", " ", key)
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]
    return None


def find_skills_in_text(text: str) -> set[str]:
    """
    Scan a block of free-form text (responsibilities, project blurbs,
    summaries) and return the set of canonical skill names whose aliases
    appear in it as whole words/phrases.

    Deterministic: same text in -> same skill set out, every time.
    """
    if not text:
        return set()

    lowered = " " + re.sub(r"\s+", " ", text.lower()) + " "
    found: set[str] = set()

    for alias in _SORTED_ALIASES:
        canonical, _category = _ALIAS_LOOKUP[alias]
        if canonical in found:
            continue
        # Word-boundary match; escape regex special chars in the alias
        # (aliases like "c++", ".net", "c#" contain regex metacharacters).
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            found.add(canonical)

    return found


def category_of(canonical_name: str) -> str:
    """Return the taxonomy category for a canonical skill name, or 'other'."""
    entry = TAXONOMY.get(canonical_name)
    return entry[0] if entry else "other"
