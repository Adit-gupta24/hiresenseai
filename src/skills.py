"""Skills Dictionary and Extraction module for HireSense AI.

Provides a comprehensive, categorized skill taxonomy, synonym/alias matching,
and configurable skill extraction from resume text and job descriptions.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Any, Tuple


# Default categorized skills taxonomy with canonical names and regex match patterns
DEFAULT_SKILLS_TAXONOMY: Dict[str, Dict[str, str]] = {
    "Programming Languages": {
        "Python": r"\bpython(?:3)?\b",
        "Java": r"\bjava\b(?!script)",
        "JavaScript": r"\b(?:javascript|js|ecmascript|es6)\b",
        "TypeScript": r"\b(?:typescript|ts)\b",
        "C++": r"\b(?:c\+\+|cplusplus|cpp)\b",
        "C#": r"\b(?:c\#|csharp)\b",
        "C": r"\b(?<!\w)c(?!\w|[+#])\b",
        "Go": r"\b(?:golang|go)\b(?!\s+(?:to|for|with))",
        "Rust": r"\brust\b",
        "Ruby": r"\bruby\b",
        "PHP": r"\bphp\b",
        "Swift": r"\bswift\b",
        "Kotlin": r"\bkotlin\b",
        "SQL": r"\bsql\b",
        "R": r"\b(?<!\w)r(?!\w|[+&])\b",
        "Scala": r"\bscala\b",
        "Dart": r"\bdart\b",
        "MATLAB": r"\bmatlab\b",
        "Bash/Shell": r"\b(?:bash|shell\s*scripting|zsh|powershell)\b",
    },
    "Web & Backend Frameworks": {
        "React": r"\b(?:react|react\.?js|reactjs)\b",
        "Angular": r"\bangular(?:\.?js)?\b",
        "Vue.js": r"\b(?:vue|vue\.?js|vuejs)\b",
        "Next.js": r"\bnext\.?js\b",
        "Node.js": r"\bnode(?:\.?js|js)?\b",
        "Express": r"\bexpress(?:\.?js)?\b",
        "Django": r"\bdjango\b",
        "FastAPI": r"\bfastapi\b",
        "Flask": r"\bflask\b",
        "Spring Boot": r"\bspring\s*boot\b",
        "ASP.NET": r"\b(?:asp\.?net|\.net(?:\s*core)?)\b",
        "Ruby on Rails": r"\b(?:rails|ruby\s+on\s+rails)\b",
        "GraphQL": r"\bgraphql\b",
        "REST API": r"\b(?:rest|restful|rest\s*apis?|web\s*apis?)\b",
        "HTML/CSS": r"\b(?:html|html5|css|css3|sass|tailwind|bootstrap)\b",
    },
    "Data Science & AI/ML": {
        "Machine Learning": r"\b(?:machine\s*learning|ml|statistical\s*learning|predictive\s*modeling)\b",
        "Deep Learning": r"\b(?:deep\s*learning|neural\s*networks?|cnn|rnn)\b",
        "NLP": r"\b(?:nlp|natural\s+language\s+processing|text\s*processing|text\s*mining)\b",
        "Computer Vision": r"\b(?:computer\s*vision|cv|image\s*processing)\b",
        "Scikit-Learn": r"\b(?:scikit[\s-]*learn|sklearn)\b",
        "TensorFlow": r"\b(?:tensorflow|tf)\b",
        "PyTorch": r"\b(?:pytorch|torch)\b",
        "Keras": r"\bkeras\b",
        "Pandas": r"\bpandas\b",
        "NumPy": r"\bnumpy\b",
        "OpenCV": r"\bopencv\b",
        "Large Language Models (LLMs)": r"\b(?:llms?|large\s+language\s+models?)\b",
        "Generative AI": r"\b(?:generative\s*ai|genai)\b",
        "Transformers": r"\b(?:transformers?|hugging\s*face|bert|gpt)\b",
        "Retrieval-Augmented Generation (RAG)": r"\b(?:rag|retrieval[\s-]augmented\s+generation)\b",
        "XGBoost": r"\bxgboost\b",
        "Data Analysis": r"\b(?:data\s+analysis|data\s+analytics|eda|exploratory\s+data\s+analysis)\b",
        "Data Visualization": r"\b(?:data\s+visualization|power\s*bi|tableau|matplotlib|seaborn|dashboarding)\b",
    },
    "Databases & Storage": {
        "PostgreSQL": r"\b(?:postgresql|postgres)\b",
        "MySQL": r"\bmysql\b",
        "MongoDB": r"\bmongodb\b",
        "Redis": r"\bredis\b",
        "SQLite": r"\bsqlite\b",
        "Oracle": r"\boracle(?:\s+db)?\b",
        "Cassandra": r"\bcassandra\b",
        "Elasticsearch": r"\belasticsearch\b",
        "DynamoDB": r"\bdynamodb\b",
        "Snowflake": r"\bsnowflake\b",
        "BigQuery": r"\bbigquery\b",
    },
    "Cloud & DevOps": {
        "AWS": r"\b(?:aws|amazon\s+web\s+services|ec2|s3)\b",
        "Azure": r"\b(?:azure|microsoft\s+azure)\b",
        "Google Cloud (GCP)": r"\b(?:gcp|google\s+cloud(?:\s+platform)?)\b",
        "Docker": r"\b(?:docker|containerization|containers?)\b",
        "Kubernetes": r"\b(?:kubernetes|k8s|container\s*orchestration)\b",
        "Terraform": r"\b(?:terraform|iac|infrastructure\s+as\s+code)\b",
        "CI/CD": r"\b(?:ci[\/\\]cd|continuous\s+integration|github\s*actions|gitlab\s*ci|jenkins)\b",
        "Linux": r"\b(?:linux|ubuntu|centos|debian|unix)\b",
        "Git": r"\b(?:git|github|gitlab|version\s*control)\b",
    },
    "Methodologies & Professional Skills": {
        "Agile/Scrum": r"\b(?:agile|scrum|kanban|sprints)\b",
        "System Design": r"\b(?:system\s+design|software\s+architecture|scalability)\b",
        "Microservices": r"\b(?:microservices?|distributed\s+systems?)\b",
        "Unit Testing": r"\b(?:unit\s+testing|pytest|unittest|tdd|automated\s+testing)\b",
        "Problem Solving": r"\bproblem\s+solving\b",
        "Team Leadership": r"\b(?:leadership|team\s+lead|mentoring)\b",
        "Communication": r"\b(?:communication\s+skills?|cross-functional\s+collaboration)\b",
        "Project Management": r"\bproject\s+management\b",
    },
}

# Extensive Alias / Synonym Dictionary for Flexible & Fuzzy Skill Matching
SKILL_ALIASES: Dict[str, List[str]] = {
    "python": ["python", "python3", "python 3", "py"],
    "pytorch": ["pytorch", "torch", "deep learning", "neural networks", "dl"],
    "tensorflow": ["tensorflow", "tf", "keras", "deep learning", "neural networks"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn", "machine learning", "ml"],
    "machine learning": ["machine learning", "ml", "statistical learning", "predictive modeling", "supervised learning", "unsupervised learning", "classification", "regression"],
    "deep learning": ["deep learning", "dl", "neural network", "neural networks", "cnn", "rnn", "transformer", "transformers", "pytorch", "tensorflow"],
    "nlp": ["nlp", "natural language processing", "text processing", "text mining", "linguistics", "language model", "transformers", "hugging face"],
    "transformers": ["transformers", "transformer", "hugging face", "bert", "gpt", "llm", "large language models"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "relational database", "rdbms", "queries", "t-sql", "pl/sql"],
    "postgresql": ["postgresql", "postgres", "sql", "relational db"],
    "mysql": ["mysql", "sql", "relational db"],
    "mongodb": ["mongodb", "nosql", "document store"],
    "redis": ["redis", "caching", "in-memory cache"],
    "docker": ["docker", "container", "containers", "containerization"],
    "kubernetes": ["kubernetes", "k8s", "container orchestration", "helm", "cluster"],
    "aws": ["aws", "amazon web services", "ec2", "s3", "cloud", "cloud computing", "lambda", "eks"],
    "azure": ["azure", "microsoft azure", "cloud"],
    "google cloud (gcp)": ["gcp", "google cloud", "google cloud platform", "bigquery"],
    "git": ["git", "github", "gitlab", "version control", "bitbucket"],
    "ci/cd": ["ci/cd", "ci-cd", "cicd", "continuous integration", "github actions", "gitlab ci", "jenkins", "devops"],
    "fastapi": ["fastapi", "rest api", "python api", "web api"],
    "flask": ["flask", "python web", "rest api"],
    "django": ["django", "python web", "django rest framework", "drf"],
    "react": ["react", "react.js", "reactjs", "frontend"],
    "node.js": ["node.js", "nodejs", "node", "express"],
    "typescript": ["typescript", "ts"],
    "javascript": ["javascript", "js", "ecmascript", "es6"],
    "data analysis": ["data analysis", "data analytics", "eda", "exploratory data analysis", "analytics"],
    "data visualization": ["data visualization", "visualization", "tableau", "power bi", "powerbi", "matplotlib", "seaborn", "dashboard"],
    "tableau": ["tableau", "data visualization", "bi dashboard"],
    "power bi": ["power bi", "powerbi", "data visualization", "bi dashboard"],
    "pandas": ["pandas", "dataframe", "data manipulation", "data analysis"],
    "numpy": ["numpy", "numerical computation"],
    "linux": ["linux", "ubuntu", "debian", "centos", "bash", "unix"],
    "rest api": ["rest api", "restful", "restful api", "restful apis", "rest apis", "web apis", "endpoints", "fastapi"],
    "microservices": ["microservices", "microservice", "distributed systems", "service-oriented"],
    "system design": ["system design", "software architecture", "scalability", "high availability"],
    "agile/scrum": ["agile", "scrum", "kanban", "sprints"],
    "unit testing": ["unit testing", "pytest", "unittest", "tdd", "automated testing", "test-driven"],
    "c++": ["c++", "cplusplus", "cpp"],
    "c#": ["c#", "csharp", "dotnet", ".net"],
    "html/css": ["html", "html5", "css", "css3", "tailwind", "tailwind css", "bootstrap", "sass"],
    "terraform": ["terraform", "iac", "infrastructure as code"],
}


def build_flat_skill_patterns(
    taxonomy: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, re.Pattern]:
    """Compiles flat lookup dictionary of canonical_skill_name -> compiled regex pattern."""
    tax = taxonomy or DEFAULT_SKILLS_TAXONOMY
    patterns: Dict[str, re.Pattern] = {}
    for _, skills in tax.items():
        for canonical_name, pattern_str in skills.items():
            patterns[canonical_name] = re.compile(pattern_str, re.IGNORECASE)
    return patterns


_FLAT_PATTERNS = build_flat_skill_patterns()


def parse_skill_string(skill_text: str) -> List[str]:
    """Parses a comma, semicolon, or newline separated list of skills entered by HR.

    Cleans up whitespace, removes duplicates while preserving order.
    """
    if not skill_text or not skill_text.strip():
        return []

    # Split by commas, semicolons, or newlines
    raw_tokens = re.split(r"[,;\n\r]+", skill_text)
    cleaned_skills: List[str] = []
    seen: Set[str] = set()

    for token in raw_tokens:
        clean_token = token.strip()
        if clean_token and clean_token.lower() not in seen:
            cleaned_skills.append(clean_token)
            seen.add(clean_token.lower())

    return cleaned_skills


def extract_skills_from_text(
    text: str,
    taxonomy: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Extracts detected skills from text against the configured taxonomy.

    Returns:
    {
        "all_skills": List[str] (canonical skill names),
        "by_category": Dict[str, List[str]] (category -> list of skills),
        "total_count": int
    }
    """
    if not text:
        return {"all_skills": [], "by_category": {}, "total_count": 0}

    tax = taxonomy or DEFAULT_SKILLS_TAXONOMY
    all_matched: Set[str] = set()
    by_category: Dict[str, List[str]] = {}

    for category, skills in tax.items():
        cat_matches: List[str] = []
        for canonical, pattern_str in skills.items():
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(text):
                cat_matches.append(canonical)
                all_matched.add(canonical)
        if cat_matches:
            by_category[category] = cat_matches

    return {
        "all_skills": sorted(list(all_matched)),
        "by_category": by_category,
        "total_count": len(all_matched),
    }


def match_target_skills(
    candidate_text: str,
    target_skills: List[str],
) -> Tuple[List[str], List[str]]:
    """Checks which target skills (required or preferred) are present in candidate text.

    Uses a three-tier matching pipeline:
    1. Exact regex boundary match for known canonical taxonomy terms.
    2. Alias / synonym matching (e.g., PyTorch matches deep learning/torch, SQL matches Postgres).
    3. Compound skill sub-component evaluation (e.g., "Python/Django" matches if candidate has either).

    Returns:
        (matched_skills, missing_skills)
    """
    if not target_skills:
        return [], []

    matched: List[str] = []
    missing: List[str] = []
    lower_text = candidate_text.lower() if candidate_text else ""

    for skill in target_skills:
        skill_clean = skill.strip()
        if not skill_clean:
            continue

        skill_lower = skill_clean.lower()
        is_matched = False

        # 1. Direct Pattern Check
        pattern = _FLAT_PATTERNS.get(skill_clean)
        if pattern is None:
            for canon, pat in _FLAT_PATTERNS.items():
                if canon.lower() == skill_lower:
                    pattern = pat
                    break

        if pattern is not None and pattern.search(candidate_text):
            is_matched = True

        # 2. Check compound skill sub-components (e.g., "PyTorch / Deep Learning" or "Python & SQL")
        if not is_matched and any(sep in skill_clean for sep in ["/", "&", "+", " or ", ","]):
            sub_parts = re.split(r"[/&+,]|\bor\b", skill_clean)
            for part in sub_parts:
                p_clean = part.strip()
                if len(p_clean) >= 2:
                    sub_pat = _FLAT_PATTERNS.get(p_clean)
                    if sub_pat and sub_pat.search(candidate_text):
                        is_matched = True
                        break
                    elif re.search(rf"\b{re.escape(p_clean.lower())}\b", lower_text):
                        is_matched = True
                        break

        # 3. Alias & Synonym dictionary matching
        if not is_matched:
            # Check direct alias key
            aliases = SKILL_ALIASES.get(skill_lower)
            if not aliases:
                # Try finding closest alias key
                for alias_key, alias_list in SKILL_ALIASES.items():
                    if alias_key in skill_lower or skill_lower in alias_key:
                        aliases = alias_list
                        break

            if aliases:
                for alias in aliases:
                    # boundary search for alias with optional plural 's'
                    if re.search(rf"\b{re.escape(alias)}(?:s)?\b", lower_text):
                        is_matched = True
                        break

        # 4. Fallback custom boundary keyword matching
        if not is_matched:
            escaped = re.escape(skill_clean)
            pattern_str = rf"(?:\b|(?<=\s)){escaped}(?:\b|(?=\s|$|[,\.;]))"
            custom_pattern = re.compile(pattern_str, re.IGNORECASE)
            if custom_pattern.search(candidate_text):
                is_matched = True

        if is_matched:
            matched.append(skill_clean)
        else:
            missing.append(skill_clean)

    return matched, missing
