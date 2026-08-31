"""Canonical skill names, aliases, and deterministic phrase matching.

Used by SkillExtractorAgent (find skills in resume text) and
SkillsMatcherAgent (compare candidate skills to JD requirements) so both
sides share one normalization scheme.
"""

from __future__ import annotations

import re
from functools import lru_cache

# (canonical display name, category, aliases including the canonical form)
# Aliases are matched after normalization; longer aliases win.
_SKILL_ENTRIES: list[tuple[str, str, tuple[str, ...]]] = [
    ("Python", "language", ("python", "python3", "py")),
    ("JavaScript", "language", ("javascript", "js", "ecmascript", "es6")),
    ("TypeScript", "language", ("typescript", "ts")),
    ("Java", "language", ("java",)),
    ("C++", "language", ("c++", "cpp", "cplusplus", "c plus plus")),
    ("C++ STL", "tool", ("c++ stl", "cpp stl", "stl")),
    ("C", "language", ("c", "c language", "c programming")),
    ("C#", "language", ("c#", "csharp", "c sharp")),
    ("Go", "language", ("golang", "go lang")),
    ("Rust", "language", ("rust",)),
    ("Ruby", "language", ("ruby",)),
    ("PHP", "language", ("php",)),
    ("R", "language", ("r language", "r programming")),
    ("Scala", "language", ("scala",)),
    ("Kotlin", "language", ("kotlin",)),
    ("Swift", "language", ("swift",)),
    ("SQL", "language", ("sql", "structured query language")),
    ("HTML", "language", ("html", "html5")),
    ("CSS", "language", ("css", "css3")),
    ("Bash", "language", ("bash", "shell scripting", "shell script")),
    ("MATLAB", "language", ("matlab",)),
    ("Machine Learning", "technical", ("machine learning", "ml")),
    ("Deep Learning", "technical", ("deep learning", "dl")),
    ("Artificial Intelligence", "technical", ("artificial intelligence", "ai")),
    ("Data Science", "technical", ("data science",)),
    ("Data Analysis", "technical", ("data analysis", "data analytics")),
    ("Natural Language Processing", "technical", ("natural language processing", "nlp")),
    ("Computer Vision", "technical", ("computer vision", "cv")),
    ("Data Structures", "technical", ("data structures", "data structure")),
    ("Algorithms", "technical", ("algorithms", "algorithm")),
    ("Data Structures & Algorithms", "technical", (
        "data structures & algorithms",
        "data structures and algorithms",
        "dsa",
        "ds and algo",
    )),
    ("Object Oriented Programming", "technical", (
        "object oriented programming",
        "object-oriented programming",
        "oop",
        "oops",
    )),
    ("Database Management Systems", "technical", (
        "database management systems",
        "database management system",
        "dbms",
    )),
    ("Database Design", "technical", ("database design", "data modeling", "schema design")),
    ("Data Warehousing", "technical", ("data warehousing", "data warehouse")),
    ("System Design", "technical", ("system design", "distributed systems")),
    ("MLOps", "technical", ("mlops",)),
    ("ETL", "technical", ("etl",)),
    ("Applied Statistics", "technical", ("applied statistics",)),
    ("Statistical Analysis", "technical", ("statistical analysis", "statistics")),
    ("Predictive Modeling", "technical", ("predictive modeling", "predictive modelling")),
    ("Recommendation Systems", "technical", ("recommendation systems", "recommender systems")),
    ("Feature Engineering", "technical", ("feature engineering",)),
    ("Model Evaluation", "technical", ("model evaluation", "model training")),
    # Common ML sub-concepts that show up as their own literal resume/JD
    # terms (e.g. "built supervised classification models", "applied
    # cross-validation", "trained regression models") but were previously
    # entirely absent from the taxonomy -- so even when a candidate's
    # resume text explicitly named them, SkillExtractorAgent could never
    # recognize them as skills at all, and they surfaced as "completely
    # missing" regardless of real textual evidence. These are each their
    # OWN canonical skill, distinct from "Machine Learning" -- adding them
    # here does NOT make "Machine Learning" satisfy any of these
    # requirements; a candidate who only lists generic ML experience with
    # no supporting text for these specific concepts still shows them as
    # gaps (see test_skill_relationship_matching.py for both directions).
    #
    # "Supervised Learning"/"Unsupervised Learning"/"Cross-validation" use
    # their full, unambiguous phrase as the alias -- no plausible
    # non-ML meaning exists for these phrases.
    ("Supervised Learning", "technical", ("supervised learning",)),
    ("Unsupervised Learning", "technical", ("unsupervised learning",)),
    ("Cross-validation", "technical", (
        "cross validation", "cross-validation", "k fold cross validation",
        "kfold cross validation", "k-fold cross validation",
    )),
    # "Classification" and "Regression" are deliberately NOT registered
    # with their bare, single-word form as a matching alias: "regression"
    # alone collides with the unrelated QA term "regression testing"
    # (common on non-ML resumes), and "classification" alone can appear
    # in non-ML contexts (e.g. data/asset classification for compliance).
    # Matching only ML-specific phrasing avoids crediting an unrelated
    # skill. The canonical DISPLAY name is still the bare word ("Regression"
    # / "Classification"), so once a resume's own ML-context phrasing is
    # recognized, the extracted skill's name exactly equals a bare
    # "Regression"/"Classification" JD requirement string and is credited
    # as an exact match via SkillsMatcherAgent's existing name-comparison
    # step -- no change needed there.
    ("Classification", "technical", (
        "classification model", "classification models", "classification algorithm",
        "classification algorithms", "classification task", "classification problem",
        "binary classification", "multi-class classification",
    )),
    ("Regression", "technical", (
        "regression model", "regression models", "regression analysis",
        "regression algorithm", "regression algorithms", "linear regression",
        "logistic regression",
    )),
    # Model-evaluation sub-concepts. A compound JD requirement like
    # "Model evaluation techniques (cross-validation, precision, recall,
    # F1-score, ROC-AUC, MAE, RMSE)" only matches a candidate's resume if
    # at least one of the individual metrics it lists is a taxonomy entry
    # that also shows up (explicitly or inferred) in the resume text --
    # before these entries existed, none of precision/recall/F1/ROC-AUC/
    # MAE/RMSE/data-preprocessing were recognized at all, so a resume
    # that clearly discussed these metrics still left the whole compound
    # requirement unmatched.
    #
    # "Precision" and "Recall" are deliberately NOT registered with their
    # bare, single-word form, for the same reason "Regression" and
    # "Classification" above aren't: standalone "precision" collides with
    # non-ML usage (precision manufacturing/instruments, "attention to
    # precision"), and standalone "recall" collides with unrelated,
    # common usage (product recall, memory recall, "I recall that...").
    # Only unambiguous, ML-metric-specific phrasing is matched --
    # including the common "precision, recall" / "precision and recall"
    # pairing used in the reported compound requirement itself, which
    # normalizes to the "precision recall" alias below.
    ("Precision", "technical", (
        "precision score", "precision metric", "model precision",
        "classification precision", "precision and recall", "precision/recall",
        "precision recall",
    )),
    ("Recall", "technical", (
        "recall score", "recall metric", "model recall",
        "classification recall", "recall and precision", "recall/precision",
    )),
    ("F1-score", "technical", (
        "f1 score", "f1-score", "f1 metric", "f-1 score", "f measure", "f1 measure",
    )),
    ("ROC-AUC", "technical", (
        "roc auc", "roc-auc", "auc roc", "roc curve", "auc score",
        "area under the curve", "area under curve",
    )),
    ("MAE", "technical", ("mean absolute error", "mae")),
    ("RMSE", "technical", (
        "root mean squared error", "root mean square error", "rmse",
    )),
    ("Data Preprocessing", "technical", (
        "data preprocessing", "data pre-processing", "data pre processing",
        "data cleaning", "data wrangling",
    )),
    ("REST APIs", "technical", ("rest apis", "rest api", "restful apis", "restful", "restful api", "rest")),
    ("GraphQL", "technical", ("graphql",)),
    ("Microservices", "technical", ("microservices", "micro services", "microservice")),
    ("Agile", "other", ("agile", "scrum", "agile scrum", "agile methodology", "agile/scrum")),
    ("CI/CD", "tool", ("ci/cd", "cicd", "ci cd", "continuous integration", "continuous deployment")),
    ("Scikit-learn", "framework", ("scikit-learn", "scikit learn", "sklearn")),
    ("TensorFlow", "framework", ("tensorflow", "tf")),
    ("PyTorch", "framework", ("pytorch", "torch")),
    ("Keras", "framework", ("keras",)),
    ("Pandas", "tool", ("pandas",)),
    ("NumPy", "tool", ("numpy", "np")),
    ("Matplotlib", "tool", ("matplotlib",)),
    ("Seaborn", "tool", ("seaborn",)),
    ("NLTK", "tool", ("nltk",)),
    ("spaCy", "tool", ("spacy",)),
    ("OpenCV", "tool", ("opencv", "open cv")),
    ("Hugging Face", "framework", ("hugging face", "huggingface", "transformers")),
    ("LangChain", "framework", ("langchain",)),
    ("Django", "framework", ("django", "django rest framework", "drf")),
    ("Flask", "framework", ("flask",)),
    ("FastAPI", "framework", ("fastapi", "fast api")),
    ("React", "framework", ("react", "reactjs", "react.js")),
    ("Angular", "framework", ("angular", "angularjs")),
    ("Vue.js", "framework", ("vue", "vuejs", "vue.js")),
    ("Node.js", "framework", ("node.js", "nodejs", "node")),
    ("Express", "framework", ("express", "expressjs", "express.js")),
    ("Spring Boot", "framework", ("spring boot", "springboot")),
    (".NET", "framework", (".net", "dotnet", "asp.net")),
    ("LangGraph", "framework", ("langgraph",)),
    ("MySQL", "tool", ("mysql",)),
    ("PostgreSQL", "tool", ("postgresql", "postgres", "psql")),
    ("MongoDB", "tool", ("mongodb", "mongo")),
    ("Redis", "tool", ("redis", "caching")),
    ("SQLite", "tool", ("sqlite",)),
    ("Oracle", "tool", ("oracle db", "oracle database")),
    ("Git", "tool", ("git", "version control")),
    ("GitHub", "tool", ("github",)),
    ("GitLab", "tool", ("gitlab",)),
    ("GitHub Actions", "tool", ("github actions",)),
    ("Google Colab", "tool", ("google colab", "colab", "colaboratory")),
    ("VS Code", "tool", ("vs code", "vscode", "visual studio code")),
    ("Arduino", "tool", ("arduino",)),
    ("Docker", "tool", ("docker", "containerization", "containers")),
    ("Kubernetes", "tool", ("kubernetes", "k8s")),
    ("Linux", "tool", ("linux",)),
    ("AWS", "tool", ("aws", "amazon web services", "ec2", "s3", "lambda", "rds", "sqs")),
    ("Azure", "tool", ("azure", "microsoft azure")),
    ("Google Cloud", "tool", ("google cloud", "gcp")),
    ("Kafka", "tool", ("kafka", "apache kafka")),
    ("Spark", "tool", ("spark", "apache spark", "pyspark")),
    ("Hadoop", "tool", ("hadoop",)),
    ("RabbitMQ", "tool", ("rabbitmq", "message queue", "message queues")),
    ("Jenkins", "tool", ("jenkins",)),
    ("Airflow", "tool", ("airflow", "apache airflow")),
    ("Heroku", "tool", ("heroku",)),
    ("Jira", "tool", ("jira",)),
    ("Snowflake", "tool", ("snowflake",)),
    ("Databricks", "tool", ("databricks",)),
    ("dbt", "tool", ("dbt", "data build tool")),
    ("Tableau", "tool", ("tableau",)),
    ("Power BI", "tool", ("power bi", "powerbi")),
    ("Excel", "tool", ("excel", "microsoft excel")),
    ("Jupyter", "tool", ("jupyter", "jupyter notebook", "jupyter notebooks")),
    ("Communication", "soft_skill", ("communication", "verbal communication", "communication skills")),
    ("Teamwork", "soft_skill", ("teamwork", "team work", "collaboration", "cross-functional", "team player")),
    ("Problem Solving", "soft_skill", ("problem solving", "problem-solving")),
    ("Leadership", "soft_skill", ("leadership", "team leadership", "led team", "mentoring", "mentored")),
    ("Code Review", "soft_skill", ("code review", "code reviews")),
]


# Tokens that are too short/ambiguous to match as standalone aliases.
_BLOCKED_STANDALONE = frozenset({"r", "go", "c", "ai", "ml", "dl", "cv", "tf", "np", "py"})

_PROTECT = [
    ("c++", "cplusplus"),
    ("c#", "csharp"),
    ("ci/cd", "cicd"),
    (".net", "dotnet"),
    ("node.js", "nodejs"),
    ("scikit-learn", "scikitlearn"),
]


def normalize_skill_text(text: str) -> str:
    """Lowercase, protect special tokens, strip punctuation to spaces."""
    if not text:
        return ""
    text = text.lower().strip()
    for src, dst in _PROTECT:
        text = text.replace(src, f" {dst} ")
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _alias_key(alias: str) -> str:
    return normalize_skill_text(alias)


@lru_cache(maxsize=1)
def _compiled_matchers() -> list[tuple[re.Pattern[str], str, str]]:
    """Patterns sorted longest-alias-first so 'c++ stl' beats 'c++' beats 'c'."""
    compiled: list[tuple[int, re.Pattern[str], str, str]] = []
    seen_alias_keys: set[str] = set()
    for canonical, category, aliases in _SKILL_ENTRIES:
        for alias in aliases:
            key = _alias_key(alias)
            if not key or key in seen_alias_keys:
                continue
            seen_alias_keys.add(key)
            if key in _BLOCKED_STANDALONE:
                continue
            pattern = re.compile(
                r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])"
            )
            compiled.append((len(key), pattern, canonical, category))

    compiled.append((2, re.compile(r"(?<![a-z0-9])ai(?![a-z0-9])"), "Artificial Intelligence", "technical"))
    compiled.append((2, re.compile(r"(?<![a-z0-9])ml(?![a-z0-9])"), "Machine Learning", "technical"))
    compiled.append((2, re.compile(r"(?<![a-z0-9])dl(?![a-z0-9])"), "Deep Learning", "technical"))
    compiled.append((1, re.compile(r"(?<![a-z0-9])c(?![a-z0-9])"), "C", "language"))
    compiled.append((1, re.compile(r"(?<![a-z0-9])r(?![a-z0-9])"), "R", "language"))
    compiled.append((2, re.compile(r"(?<![a-z0-9])go(?![a-z0-9])"), "Go", "language"))

    compiled.sort(key=lambda item: item[0], reverse=True)
    return [(pat, canon, cat) for _, pat, canon, cat in compiled]


@lru_cache(maxsize=1)
def _canonical_lookup() -> dict[str, str]:
    """Map any normalized alias or canonical name -> lowercase comparison key."""
    lookup: dict[str, str] = {}
    for canonical, _category, aliases in _SKILL_ENTRIES:
        canon_key = normalize_skill_text(canonical)
        lookup[canon_key] = canon_key
        for alias in aliases:
            lookup[_alias_key(alias)] = canon_key
    # Protected forms
    lookup["cplusplus"] = normalize_skill_text("C++")
    lookup["csharp"] = normalize_skill_text("C#")
    lookup["scikitlearn"] = normalize_skill_text("Scikit-learn")
    lookup["cicd"] = normalize_skill_text("CI/CD")
    return lookup


@lru_cache(maxsize=1)
def _display_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, _category, aliases in _SKILL_ENTRIES:
        canon_key = normalize_skill_text(canonical)
        lookup[canon_key] = canonical
        for alias in aliases:
            lookup[_alias_key(alias)] = canonical
    lookup["cplusplus"] = "C++"
    lookup["csharp"] = "C#"
    lookup["scikitlearn"] = "Scikit-learn"
    lookup["cicd"] = "CI/CD"
    return lookup


@lru_cache(maxsize=1)
def _category_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, category, aliases in _SKILL_ENTRIES:
        lookup[normalize_skill_text(canonical)] = category
        for alias in aliases:
            lookup[_alias_key(alias)] = category
    lookup["cplusplus"] = "language"
    lookup["csharp"] = "language"
    lookup["scikitlearn"] = "framework"
    lookup["cicd"] = "tool"
    return lookup


def canonical_skill_key(text: str) -> str:
    """Stable comparison key: alias-aware, lowercase, punctuation-stripped."""
    key = normalize_skill_text(text)
    return _canonical_lookup().get(key, key)


def canonical_display_name(text: str) -> str:
    key = normalize_skill_text(text)
    return _display_lookup().get(key, (text or "").strip() or text)


def skill_category(text: str) -> str:
    key = normalize_skill_text(text)
    return _category_lookup().get(key, "other")


def extract_canonical_skills(text: str) -> list[tuple[str, str]]:
    """Return unique (canonical_name, category) found in text, sorted by name."""
    if not text:
        return []
    haystack = normalize_skill_text(text)
    found: dict[str, str] = {}
    for pattern, canonical, category in _compiled_matchers():
        if pattern.search(haystack):
            found.setdefault(canonical, category)
    return sorted(found.items(), key=lambda item: item[0].casefold())


def normalize_requirement_skills(requirement: str) -> set[str]:
    """Decompose a JD requirement string into the set of canonical taxonomy
    skill keys it references.

    JobAnalyzerAgent often returns long, descriptive requirement phrases
    (e.g. "REST API development (Django REST Framework or FastAPI)" or
    "Message queues (Redis, RabbitMQ, Celery)") rather than bare skill
    tokens. Treating a whole such phrase as one opaque comparison key (as
    canonical_skill_key() does when nothing in the taxonomy matches the
    FULL string) means it can never line up with a candidate's individual
    skills. This instead scans the phrase for every taxonomy skill it
    references and returns their canonical keys individually, e.g.
    {"django", "fast api", "rest apis"}.

    Falls back to the requirement's own normalized text (single-item set)
    when nothing in the taxonomy is recognized at all, so an unrecognized
    requirement isn't silently discarded -- it just behaves as it did
    before (an opaque phrase key that can still exact-match verbatim).
    """
    hits = extract_canonical_skills(requirement)
    if hits:
        return {normalize_skill_text(name) for name, _category in hits}
    return {normalize_skill_text(requirement)}


def requirement_is_pure_soft_skill(requirement: str) -> bool:
    """True when EVERY taxonomy skill referenced by this requirement string
    is a soft_skill (e.g. "Good communication skills in English", "Strong
    teamwork and collaboration"). False for unrecognized text or for any
    requirement that references at least one non-soft-skill concept, so a
    mixed requirement like "Communication skills and Python" is still
    treated as an ordinary (scored) requirement rather than being exempted.
    """
    hits = extract_canonical_skills(requirement)
    if not hits:
        return False
    return all(category == "soft_skill" for _name, category in hits)
