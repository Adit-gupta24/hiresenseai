"""Demo Data and Sample Resumes module for HireSense AI.

Provides curated sample Job Descriptions, Presets, and realistic Candidate Resumes
for testing, grading, and presentation demonstrations.
Demo records are strictly labeled as illustrative and kept separate from real uploads.
"""

from __future__ import annotations

from typing import Dict, List, Any


# Rich Preset Roles for One-Click Evaluation & Benchmarking
JOB_PRESETS: Dict[str, Dict[str, Any]] = {
    "Senior AI / Machine Learning Engineer": {
        "job_title": "Senior AI / Machine Learning Engineer",
        "threshold": 60.0,
        "target_experience_years": 5.0,
        "target_education": "Bachelor",
        "required_skills": "Python, Machine Learning, Scikit-Learn, PyTorch, SQL, Docker, Git",
        "preferred_skills": "NLP, Transformers, FastAPI, AWS, Kubernetes",
        "description_text": """Job Title: Senior AI / Machine Learning Engineer
Department: Applied AI & Intelligent Systems
Location: Hybrid / Remote

About the Role:
We are seeking an experienced and passionate Senior AI/Machine Learning Engineer to design,
develop, and deploy scalable machine learning models and NLP systems into high-throughput production environments.
You will collaborate closely with product teams, data engineers, and software architects to deliver
actionable intelligence through predictive models, modern NLP architectures, and robust APIs.

Key Responsibilities:
- Design, build, train, and evaluate machine learning and deep learning pipelines using Python, Scikit-Learn, and PyTorch.
- Develop natural language processing (NLP) pipelines, exploring modern Transformer models, embeddings, and generative architectures.
- Clean, query, and manipulate structured datasets using SQL and relational/NoSQL databases.
- Containerize models and inference microservices using Docker, orchestrating reliable deployments with Kubernetes.
- Implement reproducible version control, code reviews, and CI/CD automation using Git and GitHub Actions.
- Optimize inference latency, monitor model drift, and collaborate on system design and cloud deployments on AWS.

Required Qualifications & Skills:
- Proficiency in Python programming and core data science libraries (NumPy, Pandas, Scikit-Learn).
- Solid theoretical understanding and practical hands-on experience with Machine Learning algorithms and Deep Learning frameworks (PyTorch or TensorFlow).
- Experience querying and managing relational databases with SQL.
- Practical experience with Docker containerization and version control using Git.
- Demonstrated ability to solve complex analytical problems and write clean, maintainable, modular code.

Preferred / Nice-to-Have Skills:
- Experience in Natural Language Processing (NLP), Hugging Face Transformers, and LLM fine-tuning or RAG pipelines.
- Experience serving models via modern web frameworks such as FastAPI or Flask.
- Familiarity with cloud platforms such as AWS (SageMaker, S3, EC2) and container orchestration with Kubernetes.
- Knowledge of automated testing (pytest) and MLOps practices.""",
    },
    "Data Analyst / BI Specialist": {
        "job_title": "Data Analyst / BI Specialist",
        "threshold": 60.0,
        "target_experience_years": 3.0,
        "target_education": "Bachelor",
        "required_skills": "SQL, Python, Pandas, Data Analysis, Data Visualization, Excel, Tableau",
        "preferred_skills": "Power BI, Statistics, R, PostgreSQL, BigQuery",
        "description_text": """Job Title: Data Analyst / BI Specialist
Department: Enterprise Analytics & Intelligence
Location: Remote

About the Role:
We are looking for a data-driven Data Analyst to transform complex transactional and operational data into clear,
actionable business insights and executive dashboards. You will work across departments to define KPIs, build
reproducible data pipelines, and craft high-impact visualizations.

Key Responsibilities:
- Write complex SQL queries, views, and CTEs to extract, clean, and aggregate data across PostgreSQL and BigQuery databases.
- Perform exploratory data analysis (EDA) and statistical evaluations using Python and Pandas.
- Design, build, and maintain interactive business intelligence dashboards in Tableau and Power BI.
- Partner with product and engineering teams to validate data accuracy and optimize reporting performance.
- Present insights, trends, and forecasts clearly to executive stakeholders.

Required Qualifications & Skills:
- Strong proficiency in SQL for data extraction, manipulation, and analysis.
- Hands-on experience with Python and Pandas for data munging and exploratory analysis.
- Proven track record of developing dashboards in Tableau or Power BI.
- Strong understanding of data visualization best practices and business metrics.

Preferred Skills:
- Experience with cloud data warehouses (BigQuery or Snowflake).
- Familiarity with statistical modeling using R or Scikit-Learn.
- Knowledge of version control using Git.""",
    },
    "Backend Software Engineer": {
        "job_title": "Backend Software Engineer",
        "threshold": 60.0,
        "target_experience_years": 4.0,
        "target_education": "Bachelor",
        "required_skills": "Python, FastAPI, Django, PostgreSQL, Docker, REST API, Git",
        "preferred_skills": "Redis, Microservices, Kubernetes, AWS, Unit Testing",
        "description_text": """Job Title: Backend Software Engineer
Department: Platform Engineering
Location: Hybrid

About the Role:
We are seeking a Backend Software Engineer to design, build, and scale resilient microservices and RESTful APIs.
You will architect database models, optimize query latency, and build reliable backend services that power
our customer-facing applications.

Key Responsibilities:
- Architect and develop high-throughput, low-latency microservices using Python, FastAPI, and Django.
- Design, optimize, and maintain relational schemas, indexes, and queries using PostgreSQL.
- Implement caching strategies and message queuing with Redis.
- Package and deploy applications using Docker containers managed on AWS cloud infrastructure.
- Write comprehensive unit and integration tests using pytest, ensuring test coverage and reliability.

Required Qualifications & Skills:
- Extensive experience with backend Python development (FastAPI or Django).
- Deep knowledge of relational database design, query optimization, and PostgreSQL.
- Solid understanding of REST API principles and microservice architectures.
- Experience with Docker containerization and Git version control.

Preferred Skills:
- Familiarity with distributed caching (Redis) and cloud deployments (AWS, Kubernetes).
- Experience with CI/CD automation pipelines and unit testing frameworks.""",
    },
    "Frontend UI/UX Engineer": {
        "job_title": "Frontend UI/UX Engineer",
        "threshold": 60.0,
        "target_experience_years": 3.0,
        "target_education": "Bachelor",
        "required_skills": "React, JavaScript, TypeScript, HTML/CSS, Next.js, Git, UI/UX Design",
        "preferred_skills": "Tailwind CSS, Vue.js, GraphQL, Figma, REST API",
        "description_text": """Job Title: Frontend UI/UX Engineer
Department: Product Experience & Design Systems
Location: Remote

About the Role:
We are looking for a creative, detail-oriented Frontend UI/UX Engineer to build beautiful, responsive, and accessible
web interfaces. You will translate wireframes and design systems from Figma into modern React and Next.js components.

Key Responsibilities:
- Build responsive, interactive web applications using React, Next.js, TypeScript, and modern CSS frameworks (Tailwind CSS).
- Collaborate with designers in Figma to create intuitive, user-friendly workflows and reusable component libraries.
- Integrate frontend views with backend REST APIs and GraphQL endpoints.
- Ensure cross-browser accessibility, responsiveness, and frontend performance optimization.
- Maintain clean, maintainable codebases with Git and agile sprint reviews.

Required Qualifications & Skills:
- Strong expertise in JavaScript, TypeScript, React, and Next.js.
- Solid mastery of modern HTML5, CSS3, and responsive design systems.
- Demonstrated passion for UI/UX principles, design aesthetics, and typography.
- Proficiency with Git version control and collaborative workflows.

Preferred Skills:
- Familiarity with Tailwind CSS, Figma, and state management tools.
- Experience consuming REST APIs and GraphQL.""",
    },
    "Full Stack Python Developer": {
        "job_title": "Full Stack Python Developer",
        "threshold": 60.0,
        "target_experience_years": 4.0,
        "target_education": "Bachelor",
        "required_skills": "Python, Django, FastAPI, React, PostgreSQL, Docker, Git",
        "preferred_skills": "TypeScript, Redis, AWS, CI/CD, Next.js",
        "description_text": """Job Title: Full Stack Python Developer
Department: Product Engineering
Location: Hybrid

About the Role:
We are seeking a versatile Full Stack Developer to build end-to-end features spanning modern React frontends
and resilient Python backends. You will work on user-facing web applications, REST APIs, and database schemas.

Key Responsibilities:
- Develop scalable backend REST APIs using Python, FastAPI, and Django.
- Build clean, interactive frontend interfaces using React, JavaScript/TypeScript, and modern CSS.
- Design and optimize relational schemas in PostgreSQL.
- Containerize full stack services with Docker and manage version control via Git.

Required Qualifications & Skills:
- Proficiency in Python web frameworks (FastAPI or Django) and frontend frameworks (React).
- Solid understanding of SQL and relational databases (PostgreSQL).
- Experience building RESTful APIs and containerizing applications with Docker.
- Strong version control practices with Git.

Preferred Skills:
- Experience with TypeScript, Next.js, Redis, and AWS cloud deployment.""",
    },
    "Cloud & DevOps Engineer": {
        "job_title": "Cloud & DevOps Engineer",
        "threshold": 60.0,
        "target_experience_years": 5.0,
        "target_education": "Bachelor",
        "required_skills": "AWS, Kubernetes, Docker, Terraform, CI/CD, Linux, Git",
        "preferred_skills": "Python, Bash/Shell, Helm, Prometheus, Microservices",
        "description_text": """Job Title: Cloud & DevOps Engineer
Department: Infrastructure & Platform Reliability
Location: Remote

About the Role:
We are looking for an experienced Cloud & DevOps Engineer to automate, scale, and secure our cloud infrastructure.
You will manage Kubernetes clusters, author Terraform infrastructure-as-code, and build automated CI/CD deployment pipelines.

Key Responsibilities:
- Architect and manage cloud infrastructure across AWS using Terraform and infrastructure-as-code principles.
- Deploy, monitor, and scale containerized services on Kubernetes (EKS) and Docker.
- Design and automate robust CI/CD pipelines using GitHub Actions or GitLab CI.
- Maintain Linux server environments, optimize networking, and implement infrastructure security policies.
- Build automation scripts in Python and Bash for health monitoring and backups.

Required Qualifications & Skills:
- Deep expertise with AWS cloud services (EC2, S3, EKS, IAM, VPC).
- Production experience with Docker containers and Kubernetes orchestration.
- Proven track record with Infrastructure-as-Code (Terraform) and Linux system administration.
- Hands-on experience building automated CI/CD pipelines and using Git.

Preferred Skills:
- Scripting proficiency with Python or Bash.
- Familiarity with Helm charts and monitoring tools like Prometheus and Grafana.""",
    },
}

# Default Sample Job Description (defaults to Senior AI/ML Engineer)
DEFAULT_SAMPLE_JD: Dict[str, Any] = JOB_PRESETS["Senior AI / Machine Learning Engineer"]


# Realistic Sample Candidates for Demo Simulation & File Generation
SAMPLE_CANDIDATES: List[Dict[str, Any]] = [
    {
        "filename": "Aris_Thorne_AI_Engineer_Resume.pdf",
        "candidate_name": "Dr. Aris Thorne",
        "email": "aris.thorne.ai@example.com",
        "phone": "+1 (555) 234-5678",
        "linkedin": "https://linkedin.com/in/aris-thorne",
        "github": "https://github.com/aristhorne",
        "page_count": 2,
        "text": """Dr. Aris Thorne
Senior Machine Learning & NLP Specialist
Email: aris.thorne.ai@example.com | Phone: +1 (555) 234-5678
LinkedIn: linkedin.com/in/aris-thorne | GitHub: github.com/aristhorne
Location: Boston, MA

PROFESSIONAL SUMMARY
Innovative Machine Learning Engineer with 6+ years of experience building scalable AI systems,
NLP pipelines, and deep learning architectures. Proven track record of taking models from research
prototypes to high-scale production using Python, Scikit-Learn, PyTorch, Docker, and AWS.

CORE TECHNICAL SKILLS
- Programming & Core: Python, SQL, C++, Bash
- ML & AI: Machine Learning, Deep Learning, NLP, Scikit-Learn, PyTorch, TensorFlow, Transformers, Hugging Face, LLMs, RAG
- Data & Databases: Pandas, NumPy, PostgreSQL, MongoDB, Redis
- Cloud & DevOps: Docker, Kubernetes, AWS, Git, CI/CD, Linux
- Frameworks & APIs: FastAPI, Flask, REST API, Microservices

WORK EXPERIENCE
Lead AI Research Engineer | Synthetix AI Labs (2021 – Present)
- Architected enterprise NLP question-answering systems leveraging Transformers and PyTorch, improving search accuracy by 34%.
- Trained custom classification, regression, and clustering algorithms using Scikit-Learn and Pandas on millions of records.
- Deployed real-time inference microservices packaged in Docker containers and managed on AWS EKS with Kubernetes.
- Mentored junior engineers, established Git workflow standards, and maintained 90%+ pytest test coverage.

Machine Learning Developer | DataVerve Technologies (2018 – 2021)
- Developed customer churn prediction models using Scikit-Learn, XGBoost, and SQL data queries.
- Built automated data preprocessing and feature extraction pipelines with Python and Pandas.
- Implemented REST API endpoints using FastAPI for internal business intelligence dashboards.

EDUCATION
Ph.D. in Computer Science (Artificial Intelligence Focus) | MIT (2018)
B.S. in Computer Science & Applied Mathematics | Carnegie Mellon University (2014)""",
    },
    {
        "filename": "Priya_Sharma_FullStack_Python_Resume.pdf",
        "candidate_name": "Priya Sharma",
        "email": "priya.sharma@example.com",
        "phone": "+1 (555) 876-5432",
        "linkedin": "https://linkedin.com/in/priyasharma-dev",
        "github": "https://github.com/priyasharma",
        "page_count": 2,
        "text": """Priya Sharma
Senior Full Stack & Python Backend Developer
Email: priya.sharma@example.com | Phone: +1 (555) 876-5432
LinkedIn: linkedin.com/in/priyasharma-dev | GitHub: github.com/priyasharma
Location: Austin, TX

PROFESSIONAL SUMMARY
Senior Software Developer with 5+ years of experience specializing in Python, Django, FastAPI,
and PostgreSQL. Adept at designing resilient REST APIs, microservices, and interactive web apps.
Solid foundational knowledge of Machine Learning concepts, data analysis, and Docker containerization.

TECHNICAL SKILLS
- Languages: Python, JavaScript, TypeScript, SQL, HTML, CSS
- Backend & Frameworks: FastAPI, Django, Flask, Node.js, REST API, GraphQL
- Databases: PostgreSQL, MySQL, Redis, SQLite
- Cloud & Tools: Docker, Git, GitHub Actions, AWS, Linux, Agile/Scrum
- Exploratory AI/Data: Pandas, NumPy, Basic Machine Learning (Scikit-Learn hobbyist)

WORK EXPERIENCE
Senior Backend Engineer | CloudScale Systems (2021 – Present)
- Designed and maintained high-performance Python microservices using FastAPI and PostgreSQL handling 2M+ daily requests.
- Integrated Docker and CI/CD pipelines through Git for automated deployment to AWS ECS.
- Collaborated with frontend team building React and TypeScript interfaces.
- Experimented with internal Scikit-Learn classification scripts for log anomaly detection.

Full Stack Developer | Nexa Solutions (2019 – 2021)
- Developed database schemas and optimized complex SQL queries across PostgreSQL databases.
- Built reusable REST APIs and internal administrative dashboards with Django and React.
- Maintained code quality through unit testing, Git version control, and peer code reviews.

EDUCATION
B.S. in Information Technology | University of Texas at Austin (2019)""",
    },
    {
        "filename": "Elena_Rostova_Cloud_DevOps_Resume.pdf",
        "candidate_name": "Elena Rostova",
        "email": "elena.rostova@example.com",
        "phone": "+1 (555) 432-1098",
        "linkedin": "https://linkedin.com/in/elena-rostova",
        "github": "https://github.com/erostova",
        "page_count": 2,
        "text": """Elena Rostova
Senior Cloud & DevOps Engineer
Email: elena.rostova@example.com | Phone: +1 (555) 432-1098
LinkedIn: linkedin.com/in/elena-rostova | GitHub: github.com/erostova
Location: Seattle, WA

PROFESSIONAL SUMMARY
Cloud and Infrastructure Engineer with 7 years of expertise in AWS, Kubernetes, Docker,
Terraform, and CI/CD automation. Skilled in Python scripting and SQL for infrastructure monitoring.

TECHNICAL EXPERTISE
- Cloud Platforms: AWS (EKS, EC2, S3, IAM, CloudWatch), Azure, Google Cloud (GCP)
- Containers & Orchestration: Docker, Kubernetes, Helm
- Infrastructure as Code: Terraform, Ansible
- CI/CD & Version Control: Git, GitHub Actions, Jenkins, GitLab CI
- Scripting & Querying: Python, Bash/Shell, SQL, Linux Administration

EXPERIENCE
Lead DevOps Engineer | Nimbus Cloud Systems (2020 – Present)
- Automated deployment of containerized services using Kubernetes, Docker, and Terraform on AWS.
- Configured robust Git CI/CD pipelines reducing deployment times by 45%.
- Wrote Python and Bash automation scripts for infrastructure health checks and SQL backups.

Site Reliability Engineer | Apex Digital (2017 – 2020)
- Managed production Linux servers, Docker clusters, and automated monitoring using Grafana.
- Spearheaded migration of legacy services to containerized architectures using Git and AWS.

EDUCATION
B.S. in Systems Engineering | University of Washington (2017)""",
    },
    {
        "filename": "Marcus_Vance_UI_Frontend_Resume.pdf",
        "candidate_name": "Marcus Vance",
        "email": "marcus.vance@example.com",
        "phone": "+1 (555) 901-2345",
        "linkedin": "https://linkedin.com/in/marcus-vance-ui",
        "github": "https://github.com/marcusvance",
        "page_count": 1,
        "text": """Marcus Vance
Frontend Developer & UI/UX Designer
Email: marcus.vance@example.com | Phone: +1 (555) 901-2345
LinkedIn: linkedin.com/in/marcus-vance-ui | GitHub: github.com/marcusvance
Location: San Francisco, CA

PROFESSIONAL SUMMARY
Creative and detail-oriented Frontend Developer with 3 years of experience crafting responsive,
accessible, and aesthetically engaging user interfaces using React, JavaScript, HTML5, and CSS3.

SKILLS & TOOLS
- Frontend: HTML/CSS, JavaScript, TypeScript, React, Next.js, Tailwind CSS, Vue.js
- Design & Prototyping: Figma, Adobe XD, UI/UX Design, Responsive Layouts
- Version Control: Git, GitHub
- Miscellaneous: Agile/Scrum, REST API consumption

EXPERIENCE
Frontend Developer | PixelCraft Interactive (2022 – Present)
- Built interactive client-facing dashboards and responsive web pages using React and Tailwind CSS.
- Collaborated with product designers in Figma to translate wireframes into clean frontend components.
- Version-controlled design systems using Git and GitHub.

Junior UI Developer | Horizon Studios (2021 – 2022)
- Implemented CSS animations, typography scales, and responsive layouts across mobile and web.
- Consumed backend REST APIs and handled frontend data state.

EDUCATION
B.A. in Digital Arts & Design | San Francisco State University (2021)""",
    },
]


def create_pdf_bytes(content: str) -> bytes:
    """Generates standard, valid PDF 1.4 bytes in-memory from plain text.

    Uses native PDF stream syntax with exact cross-reference offsets.
    Does not require heavy external PDF generation libraries.
    """
    lines = content.splitlines()

    # Build text stream
    stream_lines = ["BT", "/F1 10 Tf", "50 750 Td", "14 TL"]
    for line in lines:
        if not line.strip():
            stream_lines.append("T*")
            continue
        clean = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"({clean}) Tj")
        stream_lines.append("T*")
    stream_lines.append("ET")

    text_stream = "\n".join(stream_lines).encode("latin-1", "replace")
    stream_len = len(text_stream)

    # Build PDF objects with byte offsets
    p1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    p2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    p3 = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    p4 = f"4 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("latin-1") + text_stream + b"\nendstream\nendobj\n"
    p5 = b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"

    header = b"%PDF-1.4\n"
    o1 = len(header)
    o2 = o1 + len(p1)
    o3 = o2 + len(p2)
    o4 = o3 + len(p3)
    o5 = o4 + len(p4)
    xref_offset = o5 + len(p5)

    xref_table = f"""xref
0 6
0000000000 65535 f 
{o1:010d} 00000 n 
{o2:010d} 00000 n 
{o3:010d} 00000 n 
{o4:010d} 00000 n 
{o5:010d} 00000 n 
trailer
<<
  /Size 6
  /Root 1 0 R
>>
startxref
{xref_offset}
%%EOF
""".encode("latin-1")

    return header + p1 + p2 + p3 + p4 + p5 + xref_table


def create_scanned_empty_pdf_bytes() -> bytes:
    """Creates a valid PDF with an empty page (0 text characters) to test scanned error handling."""
    return b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer << /Size 4 /Root 1 0 R >>
startxref
190
%%EOF"""
