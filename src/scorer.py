"""Scoring and Match Analysis module for HireSense AI.

Calculates realistic, calibrated semantic relevance, alias-aware skill-match scores,
experience/education alignment, and generates human-readable explanations and recommendations.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from src.preprocessor import clean_text_for_nlp
    from src.skills import (
        extract_skills_from_text,
        match_target_skills,
        parse_skill_string,
    )
except (ImportError, ModuleNotFoundError):
    from preprocessor import clean_text_for_nlp
    from skills import (
        extract_skills_from_text,
        match_target_skills,
        parse_skill_string,
    )

# Current reference year for tenure calculations
CURRENT_YEAR = 2026


def compute_calibrated_semantic_similarity(
    resume_text: str,
    jd_text: str,
    use_transformers_if_available: bool = True,
) -> Dict[str, Any]:
    """Computes a realistic, calibrated contextual similarity score between resume and JD.

    Document-to-document raw cosine similarity in natural language is sparse (typically 0.05 - 0.25).
    This function scales raw similarity using an empirical logistic S-curve and keyword coverage
    so that matching domain candidates achieve realistic 65% - 90%+ scores rather than artificial 15%.
    """
    clean_resume = clean_text_for_nlp(resume_text, remove_stopwords=True)
    clean_jd = clean_text_for_nlp(jd_text, remove_stopwords=True)

    if not clean_resume.strip() or not clean_jd.strip():
        return {
            "semantic_score": 0.0,
            "raw_similarity": 0.0,
            "engine": "Empty Input",
            "coverage_pct": 0.0,
        }

    # 1. Check for optional sentence_transformers if installed
    if use_transformers_if_available:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb_resume = model.encode([clean_resume[:1500]])
            emb_jd = model.encode([clean_jd[:1500]])
            raw_emb_sim = float(cosine_similarity(emb_resume, emb_jd)[0][0])
            # Sentence transformer embeddings range from ~0.35 (unrelated) to ~0.85 (highly related)
            scaled_emb = max(0.0, min(100.0, (raw_emb_sim - 0.25) / 0.60 * 100.0))
            return {
                "semantic_score": round(scaled_emb, 1),
                "raw_similarity": round(raw_emb_sim, 3),
                "engine": "Sentence-Transformers (all-MiniLM-L6-v2)",
                "coverage_pct": round(scaled_emb, 1),
            }
        except Exception:
            # Graceful fallback to calibrated TF-IDF
            pass

    # 2. Lightweight, Calibrated TF-IDF + Domain Keyword Coverage
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
        raw_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])

        # Empirical Logistic S-curve calibrated for resume vs job description texts:
        # Maps raw cosine:
        # raw <= 0.04 -> 10% - 18% (Mismatched)
        # raw ~ 0.10  -> 40% - 48% (Borderline)
        # raw ~ 0.13  -> 55% - 62% (Moderate fit)
        # raw ~ 0.17  -> 72% - 80% (Strong domain alignment)
        # raw >= 0.22 -> 86% - 98% (Top match)
        k = 18.0
        x0 = 0.13
        calibrated_sim = 1.0 / (1.0 + math.exp(-k * (raw_sim - x0))) * 100.0

        # Term coverage: What percentage of high-value JD tokens appear in the resume?
        jd_words = set(w for w in clean_jd.split() if len(w) >= 3)
        res_words = set(w for w in clean_resume.split() if len(w) >= 3)
        coverage_ratio = len(jd_words & res_words) / max(1, len(jd_words))
        coverage_pct = coverage_ratio * 100.0

        # Blend calibrated similarity (75%) with token coverage (25%)
        composite_semantic = 0.75 * calibrated_sim + 0.25 * coverage_pct
        final_score = round(max(0.0, min(100.0, composite_semantic)), 1)

        return {
            "semantic_score": final_score,
            "raw_similarity": round(raw_sim, 3),
            "engine": "Calibrated Semantic NLP Engine",
            "coverage_pct": round(coverage_pct, 1),
        }
    except Exception:
        return {
            "semantic_score": 0.0,
            "raw_similarity": 0.0,
            "engine": "Fallback",
            "coverage_pct": 0.0,
        }


def compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """Legacy compatibility wrapper: returns the calibrated semantic score out of 100."""
    result = compute_calibrated_semantic_similarity(resume_text, jd_text)
    return result["semantic_score"]


def extract_experience_and_education(
    resume_text: str,
    target_years: float = 4.0,
    target_edu: str = "Bachelor",
) -> Dict[str, Any]:
    """Extracts candidate's estimated years of experience and highest education degree,

    and calculates an experience/education fit score against target criteria.
    """
    if not resume_text:
        return {
            "years_detected": 0.0,
            "edu_label": "Not Detected",
            "experience_fit_score": 50.0,
            "exp_score": 50.0,
            "edu_score": 50.0,
        }

    # 1. Experience Extraction
    exp_years = 0.0

    # Pattern A: Explicit statements (e.g. "6+ years of experience", "5 yrs experience")
    explicit_matches = re.findall(
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp|specializing|building|crafting|expertise)?",
        resume_text,
        re.I,
    )
    if explicit_matches:
        try:
            valid_nums = [float(m) for m in explicit_matches if 0.5 <= float(m) <= 40.0]
            if valid_nums:
                exp_years = max(valid_nums)
        except Exception:
            pass

    # Pattern B: Date ranges (e.g. 2018 – Present, 2019 - 2023)
    date_ranges = re.findall(
        r"\b(20\d\d|19\d\d)\s*(?:–|-|to)\s*(present|current|now|20\d\d|19\d\d)\b",
        resume_text,
        re.I,
    )
    span_years = 0.0
    if date_ranges:
        for start_str, end_str in date_ranges:
            start_yr = int(start_str)
            end_yr = CURRENT_YEAR if end_str.lower() in ["present", "current", "now"] else int(end_str)
            if 0 < (end_yr - start_yr) <= 30:
                span_years += (end_yr - start_yr)

    final_years = max(exp_years, min(span_years, 25.0))

    # Pattern C: Seniority Title Heuristic Fallback
    if final_years == 0.0:
        if re.search(r"\b(lead|principal|architect|staff|director)\b", resume_text, re.I):
            final_years = 6.0
        elif re.search(r"\bsenior\b", resume_text, re.I):
            final_years = 5.0
        elif re.search(r"\bjunior|entry\b", resume_text, re.I):
            final_years = 1.5
        else:
            final_years = 3.0

    # 2. Education Extraction
    edu_level = "Bachelor"
    edu_label = "Bachelor's Degree"
    if re.search(r"\b(ph\.?d|doctorate|doctor of philosophy)\b", resume_text, re.I):
        edu_level = "Doctorate"
        edu_label = "Ph.D. / Doctorate"
    elif re.search(r"\b(master[\'s]*|m\.?s\b|m\.?tech|m\.?sc|mba)\b", resume_text, re.I):
        edu_level = "Master"
        edu_label = "Master's Degree"
    elif re.search(r"\b(bachelor[\'s]*|b\.?s\b|b\.?tech|b\.?e\b|b\.?sc)\b", resume_text, re.I):
        edu_level = "Bachelor"
        edu_label = "Bachelor's Degree"
    elif re.search(r"\b(associate|diploma)\b", resume_text, re.I):
        edu_level = "Associate"
        edu_label = "Associate / Diploma"

    # 3. Fit Scoring
    target_y = max(1.0, float(target_years))
    if final_years >= target_y:
        exp_score = 100.0
    elif final_years >= target_y - 1.0:
        exp_score = 85.0
    elif final_years >= target_y - 2.0:
        exp_score = 70.0
    else:
        exp_score = max(35.0, (final_years / target_y) * 100.0)

    edu_hierarchy = {"Doctorate": 3, "Master": 2, "Bachelor": 1, "Associate": 0, "Any": 0}
    cand_rank = edu_hierarchy.get(edu_level, 1)
    target_rank = edu_hierarchy.get(target_edu, 1)

    if cand_rank >= target_rank:
        edu_score = 100.0
    else:
        edu_score = 80.0

    composite_fit = round(0.70 * exp_score + 0.30 * edu_score, 1)

    return {
        "years_detected": round(final_years, 1),
        "edu_label": edu_label,
        "experience_fit_score": composite_fit,
        "exp_score": round(exp_score, 1),
        "edu_score": round(edu_score, 1),
    }


def calculate_skill_match_score(
    resume_text: str,
    required_skills: List[str],
    preferred_skills: Optional[List[str]] = None,
    jd_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Calculates transparent skill-match score and identifies matched / missing skills."""
    preferred_skills = preferred_skills or []
    fallback_used = False
    fallback_note = ""

    # Check if required skills were explicitly provided
    if required_skills:
        effective_required = required_skills
    else:
        # Fallback: extract from JD text
        fallback_used = True
        jd_skills_info = extract_skills_from_text(jd_text or "")
        effective_required = jd_skills_info["all_skills"]
        if effective_required:
            fallback_note = (
                f"No required skills entered by HR; automatically benchmarked against "
                f"{len(effective_required)} skills detected in the Job Description."
            )
        else:
            fallback_note = (
                "No required skills provided and none auto-detected in the Job Description. "
                "Skill match defaulted to contextual semantic relevance."
            )

    # Match required skills (uses alias & fuzzy matching)
    matched_req, missing_req = match_target_skills(resume_text, effective_required)

    # Match preferred skills (if any)
    matched_pref, missing_pref = match_target_skills(resume_text, preferred_skills)

    # Calculate score
    if not effective_required and not preferred_skills:
        skill_score = None  # Signals caller to fall back to semantic score
    elif effective_required and not preferred_skills:
        req_ratio = len(matched_req) / max(1, len(effective_required))
        skill_score = round(req_ratio * 100.0, 1)
    elif not effective_required and preferred_skills:
        pref_ratio = len(matched_pref) / max(1, len(preferred_skills))
        skill_score = round(pref_ratio * 100.0, 1)
    else:
        # Both required and preferred present: 80% required + 20% preferred bonus
        req_ratio = len(matched_req) / max(1, len(effective_required))
        pref_ratio = len(matched_pref) / max(1, len(preferred_skills))
        composite = (0.80 * req_ratio + 0.20 * pref_ratio) * 100.0
        skill_score = round(composite, 1)

    return {
        "skill_score": skill_score,
        "effective_required": effective_required,
        "matched_required": matched_req,
        "missing_required": missing_req,
        "matched_preferred": matched_pref,
        "missing_preferred": missing_pref,
        "fallback_used": fallback_used,
        "fallback_note": fallback_note,
    }


def generate_recommendation_and_explanation(
    overall_score: float,
    threshold: float,
    matched_required: List[str],
    missing_required: List[str],
    matched_preferred: List[str],
    missing_preferred: List[str],
    semantic_score: float = 0.0,
    skill_score: float = 0.0,
    exp_fit_info: Optional[Dict[str, Any]] = None,
    # Backward compatibility with legacy keyword arguments:
    text_sim_score: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generates a structured recommendation and human-readable explanation."""
    if text_sim_score is not None:
        semantic_score = text_sim_score

    exp_fit_info = exp_fit_info or {}
    if overall_score >= threshold:
        recommendation = "Strong Match"
        rec_badge = "success"
        verdict = f"Recommended for candidate interview (Score: {overall_score:.1f}% >= threshold {threshold:.1f}%)."
    elif overall_score >= max(40.0, threshold - 15.0):
        recommendation = "Potential Match"
        rec_badge = "warning"
        verdict = f"Requires secondary HR review (Score: {overall_score:.1f}% within border range of threshold {threshold:.1f}%)."
    else:
        recommendation = "Low Match"
        rec_badge = "danger"
        verdict = f"Significant skill or contextual mismatch (Score: {overall_score:.1f}% below threshold {threshold:.1f}%)."

    # Formulate bulleted explanations
    reasons: List[str] = []

    # Semantic relevance observation
    if semantic_score >= 75.0:
        reasons.append(f"High resume-to-job contextual alignment ({semantic_score:.1f}% semantic relevance).")
    elif semantic_score >= 50.0:
        reasons.append(f"Solid vocabulary and domain overlap ({semantic_score:.1f}% semantic relevance).")
    else:
        reasons.append(f"Moderate to low contextual vocabulary overlap with the job description ({semantic_score:.1f}%).")

    # Required skills observation
    total_req = len(matched_required) + len(missing_required)
    if total_req > 0:
        req_pct = (len(matched_required) / total_req) * 100.0
        if req_pct == 100.0:
            reasons.append(f"Demonstrates 100% of required competencies ({len(matched_required)}/{total_req}): {', '.join(matched_required)}.")
        elif req_pct >= 60.0:
            reasons.append(
                f"Meets core requirements ({len(matched_required)}/{total_req} skills): {', '.join(matched_required)}. "
                f"Missing: {', '.join(missing_required)}."
            )
        else:
            reasons.append(
                f"Lacks multiple required skills ({len(missing_required)}/{total_req} missing): {', '.join(missing_required)}."
            )

    # Experience observation
    years_det = exp_fit_info.get("years_detected", 0.0)
    edu_lbl = exp_fit_info.get("edu_label", "Degree")
    reasons.append(f"Tenure & Education Profile: ~{years_det} years of relevant experience detected | {edu_lbl}.")

    # Preferred skills observation
    if matched_preferred:
        reasons.append(f"Bonus skills identified: {', '.join(matched_preferred)}.")
    elif missing_preferred:
        reasons.append(f"Missing preferred/optional skills: {', '.join(missing_preferred)}.")

    return {
        "recommendation": recommendation,
        "badge": rec_badge,
        "verdict": verdict,
        "explanation_bullets": reasons,
        "fairness_notice": (
            "Screening Indicator Notice: These scores are algorithmic estimates of resume-to-job "
            "alignment and do not confirm real-world competency. Manual human HR evaluation is required."
        ),
    }


def analyze_candidate_resume(
    resume_data: Dict[str, Any],
    job_description: str,
    required_skills: List[str],
    preferred_skills: Optional[List[str]] = None,
    threshold: float = 60.0,
    weight_skill: float = 0.50,
    weight_semantic: float = 0.40,
    weight_experience: float = 0.10,
    target_years: float = 4.0,
    target_education: str = "Bachelor",
    # Legacy alias support for backward compatibility:
    weight_text: Optional[float] = None,
) -> Dict[str, Any]:
    """Performs full end-to-end analysis on an extracted resume dictionary against a job description.

    Formula:
      Overall score = (w_skill * Skill Match) + (w_semantic * Semantic Relevance) + (w_exp * Experience Fit)
      Default weights: 50% Skill Match + 40% Semantic Relevance + 10% Experience Fit.
    """
    preferred_skills = preferred_skills or []
    resume_text = resume_data.get("text", "")

    # Handle legacy weight_text argument if passed
    if weight_text is not None and weight_text != 0.70:
        weight_semantic = weight_text

    # 1. Calibrated Semantic Relevance Score
    semantic_result = compute_calibrated_semantic_similarity(resume_text, job_description)
    semantic_score = semantic_result["semantic_score"]

    # 2. Skill Match Score (with alias matching)
    skill_result = calculate_skill_match_score(
        resume_text=resume_text,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        jd_text=job_description,
    )
    skill_score = skill_result["skill_score"]
    effective_skill_score = semantic_score if skill_score is None else skill_score

    # 3. Experience & Education Fit
    exp_fit_result = extract_experience_and_education(
        resume_text=resume_text,
        target_years=target_years,
        target_edu=target_education,
    )
    experience_score = exp_fit_result["experience_fit_score"]

    # 4. Overall Weighted Score Calculation (Normalized)
    total_weights = weight_skill + weight_semantic + weight_experience
    if total_weights <= 0:
        total_weights = 1.0

    w_skill_norm = weight_skill / total_weights
    w_sem_norm = weight_semantic / total_weights
    w_exp_norm = weight_experience / total_weights

    overall_score = round(
        (w_skill_norm * effective_skill_score)
        + (w_sem_norm * semantic_score)
        + (w_exp_norm * experience_score),
        1,
    )
    overall_score = max(0.0, min(100.0, overall_score))

    # 5. Extract all general skills found in the resume for deep profile view
    extracted_resume_skills = extract_skills_from_text(resume_text)

    # 6. Recommendation and Explanation
    rec_info = generate_recommendation_and_explanation(
        overall_score=overall_score,
        threshold=threshold,
        matched_required=skill_result["matched_required"],
        missing_required=skill_result["missing_required"],
        matched_preferred=skill_result["matched_preferred"],
        missing_preferred=skill_result["missing_preferred"],
        semantic_score=semantic_score,
        skill_score=effective_skill_score,
        exp_fit_info=exp_fit_result,
    )

    return {
        "candidate_name": resume_data.get("candidate_name", "Unknown Candidate"),
        "filename": resume_data.get("filename", "resume.pdf"),
        "email": resume_data.get("email"),
        "phone": resume_data.get("phone"),
        "linkedin": resume_data.get("linkedin"),
        "github": resume_data.get("github"),
        "page_count": resume_data.get("page_count", 0),
        "overall_score": overall_score,
        "text_similarity_score": semantic_score,  # Legacy key maintained
        "semantic_score": semantic_score,
        "raw_similarity": semantic_result["raw_similarity"],
        "semantic_engine": semantic_result["engine"],
        "skill_match_score": effective_skill_score,
        "experience_fit_score": experience_score,
        "years_experience": exp_fit_result["years_detected"],
        "education_level": exp_fit_result["edu_label"],
        "weight_skill_pct": int(w_skill_norm * 100),
        "weight_semantic_pct": int(w_sem_norm * 100),
        "weight_experience_pct": int(w_exp_norm * 100),
        "matched_required_skills": skill_result["matched_required"],
        "missing_required_skills": skill_result["missing_required"],
        "matched_preferred_skills": skill_result["matched_preferred"],
        "missing_preferred_skills": skill_result["missing_preferred"],
        "all_detected_skills": extracted_resume_skills["all_skills"],
        "skills_by_category": extracted_resume_skills["by_category"],
        "fallback_used": skill_result["fallback_used"],
        "fallback_note": skill_result["fallback_note"],
        "recommendation": rec_info["recommendation"],
        "badge": rec_info["badge"],
        "verdict": rec_info["verdict"],
        "explanation_bullets": rec_info["explanation_bullets"],
        "fairness_notice": rec_info["fairness_notice"],
        "raw_text_snippet": resume_text[:800] + ("..." if len(resume_text) > 800 else ""),
    }
