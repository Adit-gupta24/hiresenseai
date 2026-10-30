# 🎯 HireSense AI — Intelligent Automated Resume Screening & Skill Gap Analyzer

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Streamlit](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![NLP: Scikit--Learn](https://img.shields.io/badge/NLP-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, production-ready student AI application designed to empower HR professionals and recruiters to screen multiple PDF resumes against a job description, calculate transparent alignment scores, detect critical competency gaps, and export ranked candidate reports.

---

## 📌 Project Overview

Recruitment teams often receive hundreds of unstructured resumes for a single job opening. Manual screening is time-consuming, prone to human fatigue, and susceptible to unconscious bias. 

**HireSense AI** automates the preliminary screening workflow by combining **NLP text similarity (TF-IDF + Cosine Similarity)** with **deterministic, pattern-based skill matching**. It pinpoints exactly which required and preferred competencies a candidate possesses and provides actionable explanations for skill gaps — all without sending candidate data to external paid APIs.

---

## 🛠️ Technology Stack

- **Python 3.11+**
- **Streamlit**: Interactive, responsive web UI for HR screening.
- **pypdf**: Robust, local extraction of digital text from multi-page PDF resumes.
- **pandas & numpy**: Structured data handling, filtering, sorting, and CSV report generation.
- **scikit-learn**: Sublinear unigram/bigram TF-IDF vectorization and cosine similarity calculation.
- **Custom NLP Preprocessor**: Cleans stopwords, normalizes formatting, and preserves technical tokens (e.g., `C++`, `C#`, `.NET`, `Node.js`, `CI/CD`).
- **pytest & unittest**: Automated unit test suite verifying scoring formulas and edge cases.

---

## ✨ Features Breakdown

### 📊 Feature 1: Modern HR Dashboard
- **KPI Metrics Cards**: Total resumes processed, average match score, top candidate score, and count of candidates meeting or exceeding the configurable threshold.
- **Interactive Rankings Table**: Shows candidate name, source filename, overall score, text similarity %, skill match %, matched competencies, and missing skills.
- **Filtering & Search**: Filter candidates by recommendation (`Strong Match`, `Potential Match`, `Low Match`) or search by name and specific skills.
- **Exporting**: One-click download of all candidate rankings and gap details as a clean CSV report.

### 📋 Feature 2: Job Description Input & Dynamic Role Benchmarks
- Paste any job description into an expandable text area or select from **6 built-in preset roles** (*Senior AI/ML Engineer*, *Data Analyst / BI Specialist*, *Backend Software Engineer*, *Frontend UI/UX Engineer*, *Full Stack Python Developer*, *Cloud & DevOps Engineer*).
- Specify **Job Title**, **Required Skills** (mandatory), and **Preferred Skills** (nice-to-have).
- Configure the **Minimum Match Threshold** (default: `60%`).
- **Auto-Detect Skills**: Analyzes the JD text and automatically populates required skills from the built-in taxonomy.
- **Dynamic Re-scoring**: Edits to the Job Description or role presets instantly re-score all uploaded candidates in real time without requiring re-uploading files.

### 📤 Feature 3: Resume Upload & PDF Extraction
- Upload multiple PDF resumes simultaneously via drag-and-drop.
- Robust error handling for corrupted files, zero-page PDFs, and empty/scanned images.
- Clear error notification when a PDF contains no extractable digital text (recommending OCR).
- Heuristic candidate name extraction with automatic fallback to sanitized filenames.
- Automatic detection of contact information (Email, Phone number, LinkedIn, GitHub).
- In-memory cached parsing (`@st.cache_data`) for ultra-fast, snappy performance.

### 🧠 Feature 4: Hybrid Calibrated Scoring Model
Solves the traditional document sparsity issue of raw TF-IDF (which previously led to unrealistically low 10–20% match scores) with a transparent 3-pillar formula:
1. **Skill Match Score (50%)**: Direct and alias/synonym matching between candidate extracted skills and JD required/preferred competencies (with plural and compound token tolerance).
2. **Semantic / Contextual Relevance (40%)**: Sublinear TF-IDF with calibrated S-curve logistic normalization and technical domain term coverage (with optional plug-and-play `sentence-transformers` support).
3. **Experience & Education Fit (10%)**: Extracted years of experience and degree credentials evaluated against target role seniority.
- **Documented Formula**:
  $$\text{Overall Score} = (0.50 \times \text{Skill Match}) + (0.40 \times \text{Semantic Score}) + (0.10 \times \text{Experience Fit})$$
  *(Weights and pass threshold are customizable in the sidebar).*
- **Documented Fallback**: If no required skills are supplied, HireSense AI automatically extracts skills from the JD text to evaluate candidates, with a transparent note to HR.

### 🔍 Feature 5: Candidate Deep Dive & Skill Gap Analysis
- Individual candidate selector with comprehensive profile overview.
- **Demonstrated Competencies**: Green badges highlighting verified required skills and blue badges for bonus skills.
- **Identified Skill Gaps**: Red badges highlighting missing required competencies — providing specific topics for HR to probe during technical interviews.
- **Categorized Skills Taxonomy**: Visual breakdown of all detected skills across Programming Languages, Web Frameworks, AI/ML, Cloud, Databases, and Soft Skills.
- **Structured Rationale**: Human-readable bulleted explanation detailing why the candidate received their score.
- **Raw Text Inspector**: View the raw extracted digital text from the candidate's PDF.

### 🎨 Feature 6: Visual Design
- Polished navy, white, and slate blue corporate palette (`#0F172A`, `#1E3A8A`, `#2563EB`, `#F8FAFC`).
- Intuitive sidebar navigation and clean tabbed workspace.
- Clear color-coded score indicators (`High >= 70%`, `Mid 55-69%`, `Low < 55%`).

### 🚀 Feature 7: Interactive Demo Mode
- Sidebar toggle to instantly preview the application with **4 realistic benchmark candidates**:
  1. *Dr. Aris Thorne* (Senior AI/ML Engineer — Strong Match)
  2. *Priya Sharma* (Full Stack Python Developer — Potential Match)
  3. *Elena Rostova* (DevOps & Cloud Engineer — Potential Match)
  4. *Marcus Vance* (Frontend UI Developer — Low Match)
- Demo data is stored strictly in memory and labeled with clear illustrative badges, never mixed with real uploaded resumes.

### ⚖️ Feature 9: Privacy, Fairness & Ethics Charter
- **Zero Protected Attributes**: Age, gender, ethnicity, religion, caste, photographs, and names are excluded from scoring.
- **Local & Ephemeral**: Runs entirely on the local machine; resumes are never sent to external cloud APIs or persisted to a database.
- **Human-in-the-Loop**: Preliminary indicator only; automated rejections are explicitly prohibited.

---

## 📂 Project Architecture

```
aiml project/
├── app.py                      # Main Streamlit web application & UI logic
├── requirements.txt            # Pinned dependencies
├── README.md                   # Project documentation & execution guide
├── generate_sample_pdfs.py     # Generator for realistic test PDF resumes
├── src/
│   ├── __init__.py             # Package declaration
│   ├── extractor.py            # PDF parsing, metadata, contact & name extraction
│   ├── preprocessor.py         # NLP cleaning, stopwords, and token preservation
│   ├── skills.py               # Configurable skills dictionary & pattern matcher
│   ├── scorer.py               # TF-IDF cosine similarity, skill score & explanations
│   └── demo_data.py            # Benchmark sample candidates and job descriptions
├── tests/
│   ├── __init__.py             # Test package declaration
│   └── test_scorer.py          # Unit tests for scoring, skills, and extraction
└── data/
    └── sample_resumes/         # Physical sample PDF files for testing
        ├── Aris_Thorne_AI_Engineer_Resume.pdf
        ├── Priya_Sharma_FullStack_Python_Resume.pdf
        ├── Elena_Rostova_Cloud_DevOps_Resume.pdf
        ├── Marcus_Vance_UI_Frontend_Resume.pdf
        └── Scanned_Image_Empty_Resume.pdf
```

---

## ⚡ Installation & Setup Instructions

### 1. Prerequisites
Ensure Python 3.11+ is installed on your machine. You can verify your installation by running:
```powershell
python --version
```

### 2. Navigate to Project Directory
```powershell
cd "c:\Users\prime computers\OneDrive\Desktop\aiml project"
```

### 3. Install Dependencies
Install all required libraries using `pip`:
```powershell
pip install -r requirements.txt
```

*(Or use the direct python path if needed: `python -m pip install -r requirements.txt`)*

### 4. Generate Sample PDF Resumes (Optional)
If not already generated, create the realistic test PDF resumes:
```powershell
python generate_sample_pdfs.py
```

---

## 🚀 How to Launch the Application

Run the Streamlit application with the following command:

```powershell
streamlit run app.py
```

Or using python module syntax:
```powershell
python -m streamlit run app.py
```

Streamlit will launch a local web server (usually at `http://localhost:8501`) and automatically open the application in your default web browser.

---

## 🌐 Online Deployment Guide: Streamlit Community Cloud

HireSense AI is fully pre-configured for one-click deployment on **Streamlit Community Cloud** ([share.streamlit.io](https://share.streamlit.io)).

### Deployment Parameters

| Parameter | Configuration Value |
| :--- | :--- |
| **Repository** | Your GitHub repository (e.g., `your-username/hiresense-ai`) |
| **Branch** | `main` |
| **Main file path** | `app.py` *(Entry-point)* |
| **App dependencies file** | `requirements.txt` |
| **Python Version** | `3.11` or `3.12` |

### Step-by-Step GitHub & Cloud Deployment

#### Step 1: Initialize Git and Commit
From your project directory:
```bash
git init
git add .
git commit -m "Deploy HireSense AI: Automated Resume Screening & Skill Gap Analyzer"
git branch -M main
```

#### Step 2: Push to GitHub
Create a new repository on [GitHub](https://github.com/new) (e.g. `hiresense-ai`), then run:
```bash
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/hiresense-ai.git
git push -u origin main
```

#### Step 3: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
2. Click **"New app"**.
3. Fill in the deployment form:
   - **Repository:** `your-username/hiresense-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **"Deploy!"**.

Streamlit will automatically detect `requirements.txt`, install all dependencies, apply the production UI theme from `.streamlit/config.toml`, and launch your application at a public `https://<your-app-name>.streamlit.app` URL.

#### Step 4: Testing Online
- **Demo Mode**: Switch the sidebar toggle to "Activate Demo Mode" to demonstrate candidate evaluations instantly without uploading files.
- **Real PDF Uploads**: Use the expandable **"📥 Need test resumes? Download benchmark candidate PDFs"** menu in the upload area to download test candidate PDFs to your phone/PC and upload them directly.

---

## 🧪 Running Unit Tests

The test suite validates TF-IDF similarity, skill matching, edge cases, formula weights, and PDF heuristics.

To run all unit tests:
```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Or using `pytest`:
```powershell
pytest -v
```

All 16 tests run deterministically in under 0.1 seconds.

---

## 📝 Example Job Description for Testing

You can use the built-in benchmark in the app or paste this example:

```text
Job Title: Senior AI / Machine Learning Engineer
Department: Applied AI & Intelligent Systems

Responsibilities:
- Design, build, train, and evaluate machine learning and deep learning pipelines using Python, Scikit-Learn, and PyTorch.
- Develop natural language processing (NLP) pipelines, exploring modern Transformer models.
- Clean, query, and manipulate structured datasets using SQL.
- Containerize models and inference microservices using Docker and orchestrate with Kubernetes.
- Implement reproducible version control and CI/CD automation using Git.

Required Skills:
Python, Machine Learning, Scikit-Learn, PyTorch, SQL, Docker, Git

Preferred Skills:
NLP, Transformers, FastAPI, AWS, Kubernetes
```

---

## 🛡️ Privacy & Ethical AI Statement

HireSense AI does not collect, store, or sell candidate data. Resumes uploaded during a session are processed strictly in memory and are discarded when the session ends. Algorithmic scores are preliminary screening indicators and do not guarantee candidate competence. Human HR discretion is strictly required for all interview and hiring choices.
