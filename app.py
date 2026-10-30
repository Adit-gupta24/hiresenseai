"""HireSense AI — Intelligent Automated Resume Screening & Skill Gap Analyzer.

A modern, production-ready Streamlit application for HR professionals to screen multiple resumes,
calculate calibrated hybrid match scores (Skill Alignment + Semantic Relevance + Experience Fit),
identify critical skill gaps, and make informed, unbiased preliminary hiring decisions.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import streamlit as st

# Ensure project root directory is in sys.path for robust cloud imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Application modules (supports both nested src/ structure and root-level deployment)
try:
    from src.extractor import extract_text_from_pdf
    from src.preprocessor import clean_text_for_nlp
    from src.skills import (
        DEFAULT_SKILLS_TAXONOMY,
        extract_skills_from_text,
        parse_skill_string,
    )
    from src.scorer import (
        analyze_candidate_resume,
        compute_calibrated_semantic_similarity,
        extract_experience_and_education,
    )
    from src.demo_data import (
        DEFAULT_SAMPLE_JD,
        JOB_PRESETS,
        SAMPLE_CANDIDATES,
        create_pdf_bytes,
        create_scanned_empty_pdf_bytes,
    )
except (ImportError, ModuleNotFoundError):
    from extractor import extract_text_from_pdf
    from preprocessor import clean_text_for_nlp
    from skills import (
        DEFAULT_SKILLS_TAXONOMY,
        extract_skills_from_text,
        parse_skill_string,
    )
    from scorer import (
        analyze_candidate_resume,
        compute_calibrated_semantic_similarity,
        extract_experience_and_education,
    )
    from demo_data import (
        DEFAULT_SAMPLE_JD,
        JOB_PRESETS,
        SAMPLE_CANDIDATES,
        create_pdf_bytes,
        create_scanned_empty_pdf_bytes,
    )


# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="HireSense AI — Automated Resume Screening & Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Modern SaaS Custom CSS (Slate, Indigo, Emerald, Crisp Cards)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        max-width: 1280px;
    }

    /* App Header Banner */
    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #1E3A8A 100%);
        color: #FFFFFF;
        padding: 1.75rem 2.25rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15), 0 8px 10px -6px rgba(15, 23, 42, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .app-header h1 {
        color: #FFFFFF !important;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.025em;
    }
    .app-header p {
        color: #94A3B8;
        font-size: 0.96rem;
        margin: 0;
        font-weight: 500;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.12);
        color: #F8FAFC;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        margin-top: 0.85rem;
        letter-spacing: 0.02em;
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Active Job Benchmark Card */
    .job-benchmark-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .job-benchmark-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 0.35rem 0;
    }

    /* Metric KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.1rem;
        margin-bottom: 1.75rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.35rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 4px 12px -2px rgba(15, 23, 42, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(15, 23, 42, 0.08);
        border-color: #CBD5E1;
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
    }
    .kpi-title {
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        color: #0F172A;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .kpi-subtitle {
        color: #94A3B8;
        font-size: 0.78rem;
        margin-top: 0.45rem;
        font-weight: 500;
    }

    /* Candidate Result Cards */
    .candidate-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .candidate-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 6px 16px -2px rgba(15, 23, 42, 0.07);
    }
    .candidate-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    .candidate-name {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.015em;
    }
    .candidate-meta {
        color: #64748B;
        font-size: 0.84rem;
        margin-top: 0.35rem;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        align-items: center;
    }

    /* Score Badges */
    .score-badge {
        font-weight: 800;
        padding: 0.45rem 1rem;
        border-radius: 10px;
        font-size: 1.05rem;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        letter-spacing: -0.01em;
    }
    .score-high {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    .score-mid {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .score-low {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }

    /* Component breakdown chips */
    .breakdown-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.28rem 0.65rem;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #334155;
        margin-right: 0.4rem;
        margin-bottom: 0.35rem;
    }

    /* Badges / Skill Pills */
    .badge {
        display: inline-block;
        padding: 0.22rem 0.6rem;
        border-radius: 6px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        line-height: 1.35;
    }
    .badge-success {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .badge-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-info {
        background-color: #EEF2FF;
        color: #4338CA;
        border: 1px solid #E0E7FF;
    }
    .badge-neutral {
        background-color: #F1F5F9;
        color: #475569;
        border: 1px solid #E2E8F0;
    }

    /* Recommendation Alert Banner */
    .rec-banner {
        padding: 0.65rem 0.9rem;
        border-radius: 8px;
        font-size: 0.83rem;
        margin-top: 0.85rem;
        font-weight: 500;
    }
    .rec-strong {
        background: #F0FDF4;
        border-left: 4px solid #22C55E;
        color: #14532D;
    }
    .rec-potential {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        color: #78350F;
    }
    .rec-low {
        background: #FEF2F2;
        border-left: 4px solid #EF4444;
        color: #7F1D1D;
    }

    /* Demo & Ethics notice boxes */
    .demo-banner {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 0.85rem 1.25rem;
        border-radius: 0 10px 10px 0;
        color: #1E3A8A;
        font-size: 0.88rem;
        margin-bottom: 1.25rem;
    }
    .ethical-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.95rem 1.15rem;
        color: #475569;
        font-size: 0.83rem;
        line-height: 1.5;
        margin-top: 1.25rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Helper: In-memory PDF text extraction
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_pdf_bytes_cached(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Cached PDF extraction to avoid re-parsing on UI reruns."""
    return extract_text_from_pdf(file_bytes, filename=filename)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "job_title" not in st.session_state:
    st.session_state["job_title"] = DEFAULT_SAMPLE_JD["job_title"]

if "job_description" not in st.session_state:
    st.session_state["job_description"] = DEFAULT_SAMPLE_JD["description_text"]

if "required_skills" not in st.session_state:
    st.session_state["required_skills"] = DEFAULT_SAMPLE_JD["required_skills"]

if "preferred_skills" not in st.session_state:
    st.session_state["preferred_skills"] = DEFAULT_SAMPLE_JD["preferred_skills"]

if "target_experience_years" not in st.session_state:
    st.session_state["target_experience_years"] = float(DEFAULT_SAMPLE_JD.get("target_experience_years", 4.0))

if "target_education" not in st.session_state:
    st.session_state["target_education"] = DEFAULT_SAMPLE_JD.get("target_education", "Bachelor")

if "threshold" not in st.session_state:
    st.session_state["threshold"] = float(DEFAULT_SAMPLE_JD.get("threshold", 60.0))

# Calibrated Scoring Weights (Defaults: 50% Skills, 40% Semantic, 10% Experience)
if "weight_skill" not in st.session_state:
    st.session_state["weight_skill"] = 50

if "weight_semantic" not in st.session_state:
    st.session_state["weight_semantic"] = 40

if "weight_experience" not in st.session_state:
    st.session_state["weight_experience"] = 10

if "demo_mode" not in st.session_state:
    st.session_state["demo_mode"] = False

# Store raw extracted resume dictionaries so editing JD or weights re-evaluates dynamically
if "uploaded_resumes_raw" not in st.session_state:
    st.session_state["uploaded_resumes_raw"] = []

if "parsing_errors" not in st.session_state:
    st.session_state["parsing_errors"] = []

NAV_PAGES = [
    "📊 Screening Dashboard",
    "🔍 Candidate Gap Analysis",
    "📋 Job Description & Config",
    "🧠 Skills Taxonomy & Logic",
    "⚖️ Privacy, Fairness & Ethics",
]

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = NAV_PAGES[0]

# Handle programmatic page transitions (e.g. clicking 'Deep Dive')
if "target_nav" in st.session_state and st.session_state["target_nav"] in NAV_PAGES:
    st.session_state["nav_selection"] = st.session_state.pop("target_nav")


# ---------------------------------------------------------
# Dynamic Candidate Evaluation Engine
# ---------------------------------------------------------
def get_active_results() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Dynamically evaluates active candidates (either demo benchmarks or uploaded resumes)

    against the currently configured Job Description, skills, and weights.
    Edits to the JD immediately re-score without requiring re-uploading files.
    """
    req_list = parse_skill_string(st.session_state["required_skills"])
    pref_list = parse_skill_string(st.session_state["preferred_skills"])

    w_skill = st.session_state["weight_skill"] / 100.0
    w_sem = st.session_state["weight_semantic"] / 100.0
    w_exp = st.session_state["weight_experience"] / 100.0

    candidates_to_score = (
        SAMPLE_CANDIDATES
        if st.session_state["demo_mode"]
        else st.session_state["uploaded_resumes_raw"]
    )

    scored_results: List[Dict[str, Any]] = []
    for cand in candidates_to_score:
        analysis = analyze_candidate_resume(
            resume_data=cand,
            job_description=st.session_state["job_description"],
            required_skills=req_list,
            preferred_skills=pref_list,
            threshold=st.session_state["threshold"],
            weight_skill=w_skill,
            weight_semantic=w_sem,
            weight_experience=w_exp,
            target_years=st.session_state["target_experience_years"],
            target_education=st.session_state["target_education"],
        )
        scored_results.append(analysis)

    errors = [] if st.session_state["demo_mode"] else st.session_state["parsing_errors"]
    return scored_results, errors


def apply_preset(preset_key: str) -> None:
    """Applies a selected preset role into session_state and re-scores candidates."""
    if preset_key in JOB_PRESETS:
        preset = JOB_PRESETS[preset_key]
        st.session_state["job_title"] = preset["job_title"]
        st.session_state["job_description"] = preset["description_text"]
        st.session_state["required_skills"] = preset["required_skills"]
        st.session_state["preferred_skills"] = preset["preferred_skills"]
        st.session_state["target_experience_years"] = float(preset.get("target_experience_years", 4.0))
        st.session_state["target_education"] = preset.get("target_education", "Bachelor")
        st.session_state["threshold"] = float(preset.get("threshold", 60.0))


# ---------------------------------------------------------
# Sidebar: Navigation, Settings, and Mode Switcher
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 HireSense AI")
    st.caption("Intelligent Screening & Skill Gap SaaS")
    st.markdown("---")

    # Mode Selector
    st.subheader("🛠️ Operation Mode")
    demo_toggle = st.toggle(
        "Activate Demo Mode",
        value=st.session_state["demo_mode"],
        help="Simulate screening with 4 realistic benchmark resumes. Real uploads and demo data are strictly isolated.",
    )
    if demo_toggle != st.session_state["demo_mode"]:
        st.session_state["demo_mode"] = demo_toggle
        st.rerun()

    if st.session_state["demo_mode"]:
        st.info("ℹ️ **Demo Mode Active**\nShowing 4 benchmark candidates. Scores update dynamically.")

    st.markdown("---")

    # Navigation Menu
    current_nav_idx = (
        NAV_PAGES.index(st.session_state["nav_selection"])
        if st.session_state["nav_selection"] in NAV_PAGES
        else 0
    )
    nav_selection = st.radio(
        "Navigation",
        NAV_PAGES,
        index=current_nav_idx,
        key="nav_selection_radio",
    )
    st.session_state["nav_selection"] = nav_selection

    st.markdown("---")
    st.subheader("⚙️ Calibrated Scoring Model")

    # Threshold Slider (Calibrated 60% default)
    threshold_val = st.slider(
        "Minimum Pass Threshold (%)",
        min_value=35.0,
        max_value=85.0,
        value=float(st.session_state["threshold"]),
        step=5.0,
        help="Candidates scoring at or above this threshold will be flagged as 'Strong Match'.",
    )
    st.session_state["threshold"] = threshold_val

    # Formula Weight Sliders in Expander
    with st.expander("⚖️ Customize Formula Weights", expanded=False):
        st.caption("Realistic Hybrid Model: Skills + Context + Experience")
        w_sk = st.slider("Skill Match Weight (%)", 20, 70, st.session_state["weight_skill"], 5)
        w_sem = st.slider("Semantic Relevance Weight (%)", 15, 60, st.session_state["weight_semantic"], 5)
        w_exp = st.slider("Experience & Education (%)", 5, 30, st.session_state["weight_experience"], 5)

        total_w = w_sk + w_sem + w_exp
        st.caption(f"Normalized formula: **{w_sk/total_w*100:.0f}% Skills** + **{w_sem/total_w*100:.0f}% Context** + **{w_exp/total_w*100:.0f}% Experience**")
        st.session_state["weight_skill"] = w_sk
        st.session_state["weight_semantic"] = w_sem
        st.session_state["weight_experience"] = w_exp

    st.markdown("---")
    # Quick Reset
    if st.button("🔄 Clear Resumes & Reset", use_container_width=True):
        st.session_state["uploaded_resumes_raw"] = []
        st.session_state["parsing_errors"] = []
        st.rerun()


# ---------------------------------------------------------
# App Header Component
# ---------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🎯 HireSense AI</h1>
        <p>Intelligent Automated Resume Screening & Skill Gap Analyzer • Enterprise SaaS Edition</p>
        <div class="status-pill">
            <span>🟢 System Online</span> • 
            <span>⚡ Calibrated Hybrid Scoring (50/40/10)</span> • 
            <span>🛡️ Non-Biased Screening</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# TAB 1: SCREENING DASHBOARD
# ---------------------------------------------------------
if nav_selection == "📊 Screening Dashboard":
    if st.session_state["demo_mode"]:
        st.markdown(
            f"""
            <div class="demo-banner">
                <strong>💡 DEMO MODE ACTIVE:</strong> Showing 4 illustrative candidate profiles evaluated 
                against <em>"{st.session_state['job_title']}"</em>. Switch presets or edit the job description below 
                to watch rankings re-calculate dynamically in real time.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # 🎯 DYNAMIC & EDITABLE JOB DESCRIPTION PANEL
    # ---------------------------------------------------------
    with st.container():
        st.markdown(
            f"""
            <div class="job-benchmark-card">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <div style="color: #64748B; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Target Role Benchmark</div>
                        <h3 class="job-benchmark-title">{st.session_state['job_title']}</h3>
                    </div>
                    <div style="font-size: 0.82rem; color: #475569; background: #F1F5F9; padding: 0.35rem 0.75rem; border-radius: 8px;">
                        🎯 Pass Threshold: <strong>{st.session_state['threshold']:.0f}%</strong> • 
                        💼 Target Exp: <strong>{st.session_state['target_experience_years']:.0f}+ yrs</strong> • 
                        🎓 Degree: <strong>{st.session_state['target_education']}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Preset Buttons directly on the dashboard
        st.markdown("##### ⚡ Quick Switch Role Benchmark:")
        preset_cols = st.columns(len(JOB_PRESETS))
        preset_icons = {
            "Senior AI / Machine Learning Engineer": "🤖",
            "Data Analyst / BI Specialist": "📊",
            "Backend Software Engineer": "⚙️",
            "Frontend UI/UX Engineer": "🎨",
            "Full Stack Python Developer": "🌐",
            "Cloud & DevOps Engineer": "☁️",
        }
        for i, (p_name, _) in enumerate(JOB_PRESETS.items()):
            with preset_cols[i]:
                icon = preset_icons.get(p_name, "📋")
                short_label = p_name.split()[0] if len(p_name.split()) > 1 else p_name
                is_active = p_name == st.session_state["job_title"]
                btn_type = "primary" if is_active else "secondary"
                if st.button(f"{icon} {p_name}", key=f"quick_preset_{i}", type=btn_type, use_container_width=True):
                    apply_preset(p_name)
                    st.rerun()

        # Collapsible In-Place Job Description & Skills Editor
        with st.expander("✏️ Customize Job Description, Required Skills, and Target Criteria", expanded=False):
            st.caption("Changes saved here instantly re-evaluate all candidates without resetting uploads.")
            edit_c1, edit_c2 = st.columns([2, 1])
            with edit_c1:
                new_title = st.text_input("Job Title", value=st.session_state["job_title"])
                new_jd = st.text_area(
                    "Job Description Text",
                    value=st.session_state["job_description"],
                    height=200,
                    help="Paste or edit the full job specification.",
                )
            with edit_c2:
                new_exp = st.number_input(
                    "Target Min Experience (Years)",
                    min_value=0.0,
                    max_value=20.0,
                    value=float(st.session_state["target_experience_years"]),
                    step=1.0,
                )
                edu_options = ["Any", "Bachelor", "Master", "Doctorate"]
                current_edu_idx = edu_options.index(st.session_state["target_education"]) if st.session_state["target_education"] in edu_options else 1
                new_edu = st.selectbox("Target Minimum Degree", edu_options, index=current_edu_idx)

                # Auto-detect skills helper
                if st.button("🔍 Auto-Extract Skills from JD", use_container_width=True):
                    detected = extract_skills_from_text(new_jd)
                    if detected["all_skills"]:
                        st.session_state["required_skills"] = ", ".join(detected["all_skills"][:8])
                        st.session_state["preferred_skills"] = ", ".join(detected["all_skills"][8:15])
                        st.success(f"Detected {len(detected['all_skills'])} skills from JD!")
                        st.rerun()

            col_r, col_p = st.columns(2)
            with col_r:
                new_req = st.text_area("Required Skills (Comma separated)", value=st.session_state["required_skills"], height=80)
            with col_p:
                new_pref = st.text_area("Preferred / Bonus Skills (Comma separated)", value=st.session_state["preferred_skills"], height=80)

            if st.button("💾 Apply Changes & Recalculate Scores", type="primary"):
                st.session_state["job_title"] = new_title
                st.session_state["job_description"] = new_jd
                st.session_state["required_skills"] = new_req
                st.session_state["preferred_skills"] = new_pref
                st.session_state["target_experience_years"] = new_exp
                st.session_state["target_education"] = new_edu
                st.success("✅ Job benchmark updated and candidates re-scored!")
                st.rerun()

    # ---------------------------------------------------------
    # RESUME UPLOAD SECTION
    # ---------------------------------------------------------
    if not st.session_state["demo_mode"]:
        st.markdown("### 📤 Upload Candidate Resumes")

        # In-memory benchmark resume download for cloud evaluators
        with st.expander("📥 Need test resumes? Download sample benchmark PDFs", expanded=False):
            st.caption("Click below to download realistic benchmark candidate PDFs to your device, then drop them into the file uploader:")
            dl_cols = st.columns(len(SAMPLE_CANDIDATES) + 1)
            for i, cand in enumerate(SAMPLE_CANDIDATES):
                with dl_cols[i]:
                    pdf_bytes = create_pdf_bytes(cand["text"])
                    st.download_button(
                        label=f"📄 {cand['candidate_name'].split()[-1]}",
                        data=pdf_bytes,
                        file_name=cand["filename"],
                        mime="application/pdf",
                        use_container_width=True,
                        key=f"sample_dl_btn_{i}",
                    )
            with dl_cols[-1]:
                st.download_button(
                    label="📄 Scanned (Empty)",
                    data=create_scanned_empty_pdf_bytes(),
                    file_name="Scanned_Image_Empty_Resume.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="sample_dl_scanned",
                )

        uploaded_files = st.file_uploader(
            "Upload candidate PDF resumes (Multi-file supported)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload digital text-based PDF resumes. Scanned image-only PDFs will be reported.",
        )

        if uploaded_files:
            analyze_col, clear_col = st.columns([2, 1])
            with analyze_col:
                if st.button("🚀 Process & Analyze Uploads", type="primary", use_container_width=True):
                    new_raw_resumes: List[Dict[str, Any]] = []
                    new_errors: List[Dict[str, Any]] = []

                    with st.spinner("Extracting PDF text and parsing competencies..."):
                        for f in uploaded_files:
                            content_bytes = f.read()
                            extracted = parse_pdf_bytes_cached(content_bytes, f.name)
                            if not extracted["is_valid"]:
                                new_errors.append({
                                    "filename": f.name,
                                    "error": extracted.get("error", "No extractable digital text found."),
                                })
                            else:
                                new_raw_resumes.append(extracted)

                    st.session_state["uploaded_resumes_raw"] = new_raw_resumes
                    st.session_state["parsing_errors"] = new_errors
                    st.success(f"Successfully processed {len(new_raw_resumes)} resume(s)!")
                    st.rerun()

    # Retrieve Active Data dynamically evaluated
    active_candidates, active_errors = get_active_results()

    # Display Parsing / Scanned Errors if any
    if active_errors:
        st.error(f"⚠️ {len(active_errors)} file(s) could not be parsed:")
        for err in active_errors:
            st.markdown(f"- **{err['filename']}**: {err['error']}")

    if not active_candidates and not st.session_state["demo_mode"]:
        st.info(
            "👋 **No candidate resumes analyzed yet.**\n\n"
            "1. Upload PDF resumes using the file uploader above, or\n"
            "2. Turn on **'Activate Demo Mode'** in the sidebar to test immediately with 4 realistic benchmark candidate profiles."
        )
    else:
        # Sort candidates by overall score descending
        sorted_candidates = sorted(active_candidates, key=lambda c: c["overall_score"], reverse=True)
        total_resumes = len(sorted_candidates)
        avg_score = sum(c["overall_score"] for c in sorted_candidates) / total_resumes if total_resumes else 0.0
        top_score = sorted_candidates[0]["overall_score"] if sorted_candidates else 0.0
        top_candidate_name = sorted_candidates[0]["candidate_name"] if sorted_candidates else "N/A"
        above_threshold = sum(1 for c in sorted_candidates if c["overall_score"] >= st.session_state["threshold"])
        pass_rate = (above_threshold / total_resumes * 100.0) if total_resumes else 0.0

        # KPI Metrics Cards
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-title">Total Resumes</div>
                    <div class="kpi-value">{total_resumes}</div>
                    <div class="kpi-subtitle">{"Demo Benchmarks" if st.session_state['demo_mode'] else "Parsed Documents"}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Average Match</div>
                    <div class="kpi-value">{avg_score:.1f}%</div>
                    <div class="kpi-subtitle">Hybrid Calibrated Score</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Qualified Candidates</div>
                    <div class="kpi-value" style="color: #10B981;">{above_threshold}</div>
                    <div class="kpi-subtitle">{pass_rate:.0f}% Pass Rate (&ge; {st.session_state['threshold']:.0f}%)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Top Candidate</div>
                    <div class="kpi-value" style="color: #4F46E5;">{top_score:.1f}%</div>
                    <div class="kpi-subtitle">{top_candidate_name}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Filters, Search, and Sort Controls
        st.markdown("### 📋 Ranked Candidate Shortlist")
        f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 1.5])

        with f_col1:
            rec_filter = st.selectbox(
                "Filter Recommendation",
                ["All Candidates", "Strong Match", "Potential Match", "Low Match"],
            )

        with f_col2:
            sort_by = st.selectbox(
                "Sort Candidates By",
                ["Overall Match Score", "Skill Match Score", "Semantic Context", "Experience Fit"],
            )

        with f_col3:
            search_query = st.text_input("Search Candidate or Skill", placeholder="e.g. Python, Aris, SQL...")

        with f_col4:
            # CSV Download
            csv_rows = []
            for c in sorted_candidates:
                csv_rows.append({
                    "Rank": 0,
                    "Candidate Name": c["candidate_name"],
                    "Overall Match (%)": c["overall_score"],
                    "Skill Match (%)": c["skill_match_score"],
                    "Semantic Context (%)": c["semantic_score"],
                    "Experience Fit (%)": c["experience_fit_score"],
                    "Experience Detected": f"{c.get('years_experience', 0)} yrs",
                    "Education": c.get("education_level", ""),
                    "Recommendation": c["recommendation"],
                    "Matching Skills": "; ".join(c["matched_required_skills"] + c["matched_preferred_skills"]),
                    "Missing Required Skills": "; ".join(c["missing_required_skills"]),
                    "Missing Preferred Skills": "; ".join(c["missing_preferred_skills"]),
                    "Email": c.get("email", ""),
                    "Phone": c.get("phone", ""),
                })
            df_export = pd.DataFrame(csv_rows)
            for r_idx, row in enumerate(csv_rows):
                row["Rank"] = r_idx + 1
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name=f"hiresense_screening_{st.session_state['job_title'].replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Apply sorting
        if sort_by == "Skill Match Score":
            sorted_candidates = sorted(sorted_candidates, key=lambda c: c["skill_match_score"], reverse=True)
        elif sort_by == "Semantic Context":
            sorted_candidates = sorted(sorted_candidates, key=lambda c: c["semantic_score"], reverse=True)
        elif sort_by == "Experience Fit":
            sorted_candidates = sorted(sorted_candidates, key=lambda c: c["experience_fit_score"], reverse=True)
        else:
            sorted_candidates = sorted(sorted_candidates, key=lambda c: c["overall_score"], reverse=True)

        # Filter by Recommendation
        filtered = sorted_candidates
        if rec_filter != "All Candidates":
            filtered = [c for c in filtered if c["recommendation"] == rec_filter]

        # Filter by Search
        if search_query.strip():
            sq = search_query.strip().lower()
            filtered = [
                c for c in filtered
                if sq in c["candidate_name"].lower()
                or sq in c["filename"].lower()
                or any(sq in s.lower() for s in c["matched_required_skills"] + c["matched_preferred_skills"])
            ]

        # Display Candidate Cards
        if not filtered:
            st.warning("No candidates matched your filter or search criteria.")
        else:
            for idx, c in enumerate(filtered):
                score_class = (
                    "score-high" if c["overall_score"] >= st.session_state["threshold"]
                    else "score-mid" if c["overall_score"] >= st.session_state["threshold"] - 15
                    else "score-low"
                )
                rec_class = (
                    "rec-strong" if c["recommendation"] == "Strong Match"
                    else "rec-potential" if c["recommendation"] == "Potential Match"
                    else "rec-low"
                )

                with st.container():
                    st.markdown(
                        f"""
                        <div class="candidate-card">
                            <div class="candidate-header">
                                <div>
                                    <div style="display: flex; align-items: center; gap: 8px;">
                                        <span style="background: #0F172A; color: #FFF; font-size: 0.75rem; font-weight: 700; padding: 2px 8px; border-radius: 6px;">#{idx + 1}</span>
                                        <h3 class="candidate-name">{c['candidate_name']}</h3>
                                    </div>
                                    <div class="candidate-meta">
                                        <span>📁 <code>{c['filename']}</code></span>
                                        <span>💼 <strong>{c.get('years_experience', 0):.1f} yrs</strong> experience</span>
                                        <span>🎓 <strong>{c.get('education_level', 'Degree')}</strong></span>
                                        <span>✉️ {c['email'] or 'Email unlisted'}</span>
                                        <span>📞 {c['phone'] or 'Phone unlisted'}</span>
                                    </div>
                                </div>
                                <div style="text-align: right;">
                                    <div class="score-badge {score_class}">{c['overall_score']:.1f}%</div>
                                    <div style="font-size: 0.8rem; font-weight: 700; color: #475569; margin-top: 0.3rem;">
                                        {c['recommendation']}
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Score Breakdown Columns
                    sc1, sc2, sc3 = st.columns([2, 3.2, 1.2])

                    with sc1:
                        st.markdown("**Hybrid Score Breakdown:**")
                        st.progress(min(1.0, c["overall_score"] / 100.0))
                        st.markdown(
                            f"""
                            <div style="margin-top: 0.4rem;">
                                <span class="breakdown-chip">🎯 Skills: <strong>{c['skill_match_score']:.1f}%</strong> ({c['weight_skill_pct']}%)</span>
                                <span class="breakdown-chip">🧠 Context: <strong>{c['semantic_score']:.1f}%</strong> ({c['weight_semantic_pct']}%)</span>
                                <span class="breakdown-chip">💼 Exp & Edu: <strong>{c['experience_fit_score']:.1f}%</strong> ({c['weight_experience_pct']}%)</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with sc2:
                        st.markdown("**Competency Gap Overview:**")
                        matched_html = "".join([f'<span class="badge badge-success">✓ {s}</span>' for s in c["matched_required_skills"]])
                        if not matched_html:
                            matched_html = '<span style="color:#94A3B8; font-size:0.8rem;">None detected</span>'
                        st.markdown(f"**Matched Required:** {matched_html}", unsafe_allow_html=True)

                        missing_html = "".join([f'<span class="badge badge-danger">✗ {s}</span>' for s in c["missing_required_skills"]])
                        if not missing_html:
                            missing_html = '<span class="badge badge-neutral">All Required Skills Verified</span>'
                        st.markdown(f"**Missing Required:** {missing_html}", unsafe_allow_html=True)

                        if c["matched_preferred_skills"]:
                            pref_matched_html = "".join([f'<span class="badge badge-info">★ {s}</span>' for s in c["matched_preferred_skills"]])
                            st.markdown(f"**Bonus Skills:** {pref_matched_html}", unsafe_allow_html=True)

                    with sc3:
                        st.write("")
                        if st.button("🔍 Deep Dive", key=f"btn_dive_{idx}_{c['filename']}", use_container_width=True):
                            st.session_state["selected_candidate_name"] = c["candidate_name"]
                            st.session_state["target_nav"] = "🔍 Candidate Gap Analysis"
                            st.rerun()

                    # Actionable HR Recommendation Box
                    st.markdown(
                        f"""
                        <div class="rec-banner {rec_class}">
                            <strong>Recommendation:</strong> {c['verdict']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")

        # Screening Indicator Notice
        st.markdown(
            """
            <div class="ethical-box">
                ⚖️ <strong>Decision Support Disclaimer:</strong> HireSense AI calculates preliminary candidate-to-job 
                alignment using a calibrated hybrid model (Skill Taxonomy Match + Semantic Relevance + Experience Fit). 
                Automated scores are intended to assist human HR prioritization; all hiring decisions must be finalized 
                by qualified recruiters following structured interviews.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# TAB 2: CANDIDATE DEEP DIVE & SKILL GAP ANALYSIS
# ---------------------------------------------------------
elif nav_selection == "🔍 Candidate Gap Analysis":
    active_candidates, _ = get_active_results()

    if not active_candidates:
        st.info("Please upload resumes or activate Demo Mode in the sidebar first to view detailed gap analysis.")
    else:
        top_c1, _ = st.columns([1.5, 4])
        with top_c1:
            if st.button("⬅️ Back to Rankings Shortlist", use_container_width=True):
                st.session_state["target_nav"] = "📊 Screening Dashboard"
                st.rerun()

        candidate_names = [c["candidate_name"] for c in active_candidates]
        default_idx = 0
        if "selected_candidate_name" in st.session_state:
            if st.session_state["selected_candidate_name"] in candidate_names:
                default_idx = candidate_names.index(st.session_state["selected_candidate_name"])

        selected_name = st.selectbox(
            "Select Candidate for In-Depth Evaluation",
            candidate_names,
            index=default_idx,
        )

        candidate = next(c for c in active_candidates if c["candidate_name"] == selected_name)

        # Candidate Profile Card
        st.markdown(
            f"""
            <div class="candidate-card">
                <div class="candidate-header">
                    <div>
                        <h2 class="candidate-name">{candidate['candidate_name']}</h2>
                        <div class="candidate-meta" style="margin-top: 0.4rem;">
                            <span>📁 Source File: <code>{candidate['filename']}</code></span>
                            <span>📄 Page Count: {candidate['page_count']}</span>
                            <span>💼 Tenure: <strong>~{candidate.get('years_experience', 0):.1f} years</strong></span>
                            <span>🎓 Education: <strong>{candidate.get('education_level', 'Degree')}</strong></span>
                        </div>
                        <div class="candidate-meta" style="margin-top: 0.4rem;">
                            <span>✉️ <strong>Email:</strong> {candidate['email'] or 'Not detected'}</span>
                            <span>📞 <strong>Phone:</strong> {candidate['phone'] or 'Not detected'}</span>
                            <span>🔗 <strong>LinkedIn:</strong> {candidate['linkedin'] or 'Not detected'}</span>
                            <span>💻 <strong>GitHub:</strong> {candidate['github'] or 'Not detected'}</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.4rem; font-weight: 800; color: #1E3A8A; line-height: 1;">
                            {candidate['overall_score']:.1f}%
                        </div>
                        <span class="badge badge-info" style="margin-top: 0.35rem;">{candidate['recommendation']}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrics Breakdown
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Overall Score", f"{candidate['overall_score']:.1f}%", delta=f"{candidate['overall_score'] - st.session_state['threshold']:.1f}% vs threshold")
        with m2:
            st.metric("Skill Alignment", f"{candidate['skill_match_score']:.1f}%", help=f"Weighted at {candidate['weight_skill_pct']}%")
        with m3:
            st.metric("Semantic Context", f"{candidate['semantic_score']:.1f}%", help=f"Weighted at {candidate['weight_semantic_pct']}%\nEngine: {candidate.get('semantic_engine', 'NLP')}")
        with m4:
            st.metric("Experience & Edu Fit", f"{candidate['experience_fit_score']:.1f}%", help=f"Weighted at {candidate['weight_experience_pct']}%")

        st.markdown("### 📊 Competency & Skill Gap Analysis")

        gap_col1, gap_col2 = st.columns(2)

        with gap_col1:
            st.markdown("#### ✅ Demonstrated Competencies")
            if candidate["matched_required_skills"]:
                st.markdown("**Core Required Skills Verified:**")
                for s in candidate["matched_required_skills"]:
                    st.markdown(f"- <span class='badge badge-success'>✓ {s}</span>", unsafe_allow_html=True)
            else:
                st.markdown("*No required skills detected in resume.*")

            if candidate["matched_preferred_skills"]:
                st.markdown("**Bonus / Preferred Skills Verified:**")
                for s in candidate["matched_preferred_skills"]:
                    st.markdown(f"- <span class='badge badge-info'>★ {s}</span>", unsafe_allow_html=True)

        with gap_col2:
            st.markdown("#### ⚠️ Identified Skill Gaps & Interview Topics")
            if candidate["missing_required_skills"]:
                st.markdown("**Missing Required Skills (Recommended Topics to Probe in Interview):**")
                for s in candidate["missing_required_skills"]:
                    st.markdown(f"- <span class='badge badge-danger'>✗ Missing: {s}</span> *(Ask candidate about hands-on exposure)*", unsafe_allow_html=True)
            else:
                st.success("🎉 Zero required skill gaps detected! Candidate demonstrates all mandatory competencies.")

            if candidate["missing_preferred_skills"]:
                st.markdown("**Missing Preferred Skills:**")
                for s in candidate["missing_preferred_skills"]:
                    st.markdown(f"- <span class='badge badge-warning'>− Missing: {s}</span>", unsafe_allow_html=True)

        st.markdown("---")

        # Full Skills Profile by Category
        st.markdown("#### 🗂️ All Skills Detected on Candidate's Resume (Taxonomy Breakdown)")
        if candidate.get("skills_by_category"):
            cat_keys = list(candidate["skills_by_category"].keys())
            cat_cols = st.columns(min(4, len(cat_keys)))
            for idx, cat_name in enumerate(cat_keys):
                cat_skills = candidate["skills_by_category"][cat_name]
                with cat_cols[idx % len(cat_cols)]:
                    st.markdown(f"**{cat_name}:**")
                    for sk in cat_skills:
                        st.markdown(f"<span class='badge badge-neutral'>{sk}</span>", unsafe_allow_html=True)
        else:
            st.info("No standard categorized skills detected in resume text.")

        st.markdown("---")

        # Recommendation & Rationale
        st.markdown("#### 📝 Algorithmic Evaluation & Screening Rationale")
        st.info(f"**Recommendation Verdict:** {candidate['verdict']}")
        for bullet in candidate["explanation_bullets"]:
            st.markdown(f"- {bullet}")

        if candidate.get("fallback_used"):
            st.warning(f"ℹ️ {candidate.get('fallback_note')}")

        # Collapsible Raw Resume Text Inspector
        with st.expander("📄 View Extracted Raw Resume Text"):
            st.text_area("Cleaned / Extracted Content", candidate.get("raw_text_snippet", ""), height=250)


# ---------------------------------------------------------
# TAB 3: JOB DESCRIPTION & CONFIGURATION
# ---------------------------------------------------------
elif nav_selection == "📋 Job Description & Config":
    st.markdown("### 📋 Configure Job Benchmark Requirements")
    st.caption("Customize the job title, requirements, and skill definitions used to evaluate resumes.")

    st.markdown("#### ⚡ 1-Click Role Presets")
    preset_cols = st.columns(len(JOB_PRESETS))
    for i, (p_name, p_data) in enumerate(JOB_PRESETS.items()):
        with preset_cols[i]:
            if st.button(f"📋 {p_name}", key=f"tab_preset_{i}", use_container_width=True):
                apply_preset(p_name)
                st.success(f"Applied preset: {p_name}")
                st.rerun()

    st.markdown("---")
    st.markdown("#### 🛠️ Custom Job Specification")
    c1, c2 = st.columns([2, 1])
    with c1:
        job_title_input = st.text_input("Job Title", value=st.session_state["job_title"])
        jd_text_input = st.text_area(
            "Job Description Text",
            value=st.session_state["job_description"],
            height=260,
            help="Paste the full job specification here. The text will be processed for semantic relevance comparison.",
        )
    with c2:
        exp_input = st.number_input(
            "Target Minimum Experience (Years)",
            min_value=0.0,
            max_value=25.0,
            value=float(st.session_state["target_experience_years"]),
            step=1.0,
        )
        edu_list = ["Any", "Bachelor", "Master", "Doctorate"]
        curr_idx = edu_list.index(st.session_state["target_education"]) if st.session_state["target_education"] in edu_list else 1
        edu_input = st.selectbox("Target Minimum Degree Level", edu_list, index=curr_idx)

        st.caption("Auto-Extraction Helper:")
        if st.button("🔍 Auto-Detect Skills from JD", use_container_width=True):
            detected = extract_skills_from_text(jd_text_input)
            if detected["all_skills"]:
                st.session_state["required_skills"] = ", ".join(detected["all_skills"][:8])
                st.session_state["preferred_skills"] = ", ".join(detected["all_skills"][8:15])
                st.success(f"Detected {len(detected['all_skills'])} skills from JD text!")
                st.rerun()
            else:
                st.warning("No standard skills detected from the taxonomy in this JD text.")

    col_req, col_pref = st.columns(2)
    with col_req:
        req_skills_input = st.text_area(
            "Required Skills (Comma or Newline Separated)",
            value=st.session_state["required_skills"],
            height=100,
            help="Mandatory competencies. Missing these reduces the candidate's skill score significantly.",
        )
    with col_pref:
        pref_skills_input = st.text_area(
            "Preferred / Optional Skills (Comma Separated)",
            value=st.session_state["preferred_skills"],
            height=100,
            help="Bonus skills that provide competitive advantage in ranking.",
        )

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("💾 Save Job Configuration", type="primary", use_container_width=True):
            st.session_state["job_title"] = job_title_input
            st.session_state["job_description"] = jd_text_input
            st.session_state["required_skills"] = req_skills_input
            st.session_state["preferred_skills"] = pref_skills_input
            st.session_state["target_experience_years"] = exp_input
            st.session_state["target_education"] = edu_input
            st.success("✅ Job benchmark successfully updated!")
    with btn_col2:
        if st.button("📊 Save & Return to Dashboard", use_container_width=True):
            st.session_state["job_title"] = job_title_input
            st.session_state["job_description"] = jd_text_input
            st.session_state["required_skills"] = req_skills_input
            st.session_state["preferred_skills"] = pref_skills_input
            st.session_state["target_experience_years"] = exp_input
            st.session_state["target_education"] = edu_input
            st.session_state["target_nav"] = "📊 Screening Dashboard"
            st.rerun()


# ---------------------------------------------------------
# TAB 4: SKILLS TAXONOMY & MATHEMATICAL FORMULAS
# ---------------------------------------------------------
elif nav_selection == "🧠 Skills Taxonomy & Logic":
    st.markdown("### 🧠 Built-In Skills Taxonomy & Scoring Engine")
    st.caption(
        "HireSense AI uses an extensible, case-insensitive taxonomy with regex boundaries, "
        "synonym/alias mapping, and technical token preservation (e.g. C++, .NET, CI/CD)."
    )

    for cat_name, skills_dict in DEFAULT_SKILLS_TAXONOMY.items():
        with st.expander(f"📁 {cat_name} ({len(skills_dict)} skills)", expanded=False):
            skills_list = list(skills_dict.keys())
            cols = st.columns(4)
            for i, skill in enumerate(skills_list):
                with cols[i % 4]:
                    st.markdown(f"- `{skill}`")

    st.markdown("---")
    st.markdown("#### 📐 Calibrated Hybrid Scoring Model")
    st.markdown(
        r"""
        HireSense AI employs a 3-pillar, realistic hybrid matching algorithm:

        1. **Skill Alignment Score ($S_{\text{skill}}$, 50% default weight)**:
           $$S_{\text{skill}} = \left( 0.80 \times \frac{|\text{Matched}_{\text{req}}|}{|\text{Required}|} + 0.20 \times \frac{|\text{Matched}_{\text{pref}}|}{|\text{Preferred}|} \right) \times 100$$
           - Utilizes alias and synonym matching (e.g., PyTorch matches deep learning/torch, SQL matches Postgres, React matches React.js).
           - Compound skill matching (e.g., "Python/Django" rewards either competency).

        2. **Semantic / Contextual Relevance ($S_{\text{semantic}}$, 40% default weight)**:
           - Raw TF-IDF cosine similarity between full documents naturally clusters between $0.05$ and $0.25$.
           - HireSense AI applies an empirical logistic S-curve calibration blended with domain term coverage:
             $$S_{\text{calibrated}} = \frac{1}{1 + e^{-18(s_{\text{raw}} - 0.13)}} \times 100$$
           - Supported with `sentence-transformers` (`all-MiniLM-L6-v2`) when available.

        3. **Experience & Education Fit ($S_{\text{experience}}$, 10% default weight)**:
           - Automatically extracts candidate tenure (explicit years + career date spans) and highest degree.
           - Evaluates candidate fit against target minimum tenure and degree level.

        4. **Overall Composite Score**:
           $$S_{\text{overall}} = (w_{\text{skill}} \times S_{\text{skill}}) + (w_{\text{semantic}} \times S_{\text{semantic}}) + (w_{\text{experience}} \times S_{\text{experience}})$$
        """
    )


# ---------------------------------------------------------
# TAB 5: PRIVACY, FAIRNESS & ETHICS
# ---------------------------------------------------------
elif nav_selection == "⚖️ Privacy, Fairness & Ethics":
    st.markdown("### ⚖️ AI Ethics, Privacy & Fairness Charter")
    st.caption("HireSense AI is designed with responsible, transparent, and non-discriminatory HR technology principles.")

    st.markdown(
        """
        #### 1. Non-Discrimination Policy
        - **Protected Attributes Excluded:** HireSense AI explicitly **does not** extract, parse, or evaluate candidate age, gender, race, ethnicity, religion, caste, nationality, physical ability, sexual orientation, or photographs.
        - **Name Sanitization:** Candidate names are extracted strictly for human HR indexing and identification. Algorithms never assign points, weights, or ranking variations based on names.

        #### 2. Local & Ephemeral Processing
        - **Zero Cloud API Reliance:** Core text processing runs locally using Scikit-Learn TF-IDF, PyPDF, and deterministic regex.
        - **No Third-Party Transmission:** Candidate resumes are processed strictly in-memory during the active browser session. They are **never** transmitted to paid external AI APIs or remote storage endpoints.
        - **No Permanent Retention:** Uploaded files and in-memory caches are purged upon session termination or clicking the "Reset Analysis" button.

        #### 3. Human-in-the-Loop Imperative
        - **No Automated Rejection:** HireSense AI is strictly a preliminary candidate screening and prioritization aid. It does not issue automatic rejection emails or hiring decisions.
        - **Contextual Awareness:** Keyword matching does not equal real-world competence. HR professionals must manually review borderline candidates and conduct structured competency interviews.

        #### 4. Auditability & Explainability
        - Every score is backed by an itemized, human-readable breakdown detailing:
          - Skill match score and specific demonstrated vs missing skills.
          - Semantic context similarity percentage.
          - Experience and tenure alignment.
          - Transparent fallback notes whenever inputs are missing.
        """
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "HireSense AI • Built with Python, Streamlit, Scikit-Learn, and PyPDF • "
    "Designed for Student AI / HR Analytics Demonstration."
)
