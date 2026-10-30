"""Unit tests for HireSense AI scoring logic, skill extraction, and preprocessing."""

import unittest
try:
    from src.preprocessor import clean_text_for_nlp
    from src.skills import (
        extract_skills_from_text,
        match_target_skills,
        parse_skill_string,
    )
    from src.scorer import (
        compute_tfidf_similarity,
        calculate_skill_match_score,
        analyze_candidate_resume,
        generate_recommendation_and_explanation,
    )
    from src.extractor import sanitize_filename_to_name, extract_candidate_name_heuristic
except (ImportError, ModuleNotFoundError):
    from preprocessor import clean_text_for_nlp
    from skills import (
        extract_skills_from_text,
        match_target_skills,
        parse_skill_string,
    )
    from scorer import (
        compute_tfidf_similarity,
        calculate_skill_match_score,
        analyze_candidate_resume,
        generate_recommendation_and_explanation,
    )
    from extractor import sanitize_filename_to_name, extract_candidate_name_heuristic


class TestPreprocessor(unittest.TestCase):
    def test_clean_text_removes_stopwords_and_urls(self):
        text = "Candidate Alice profile at https://github.com/dev and email at dev@example.com with 5 years experience."
        cleaned = clean_text_for_nlp(text, remove_stopwords=True)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("example.com", cleaned)
        self.assertNotIn("candidate", cleaned)
        self.assertIn("alice", cleaned)

    def test_clean_text_preserves_tech_tokens(self):
        text = "Skilled in C++, C#, .NET, and Node.js for backend systems."
        cleaned = clean_text_for_nlp(text)
        self.assertIn("cplusplus", cleaned)
        self.assertIn("csharp", cleaned)
        self.assertIn("dotnet", cleaned)
        self.assertIn("nodejs", cleaned)


class TestSkillExtraction(unittest.TestCase):
    def test_parse_skill_string(self):
        raw = "Python, Machine Learning; Docker\nKubernetes, python"
        parsed = parse_skill_string(raw)
        self.assertEqual(parsed, ["Python", "Machine Learning", "Docker", "Kubernetes"])

    def test_extract_skills_from_text(self):
        text = "Candidate specializes in Python, Scikit-Learn, and PostgreSQL on AWS."
        result = extract_skills_from_text(text)
        all_skills = result["all_skills"]
        self.assertIn("Python", all_skills)
        self.assertIn("Scikit-Learn", all_skills)
        self.assertIn("PostgreSQL", all_skills)
        self.assertIn("AWS", all_skills)

    def test_match_target_skills_case_insensitive(self):
        resume_text = "Proficient in python, DOCKER, and kubernetes."
        target = ["Python", "Docker", "Kubernetes", "Rust"]
        matched, missing = match_target_skills(resume_text, target)
        self.assertIn("Python", matched)
        self.assertIn("Docker", matched)
        self.assertIn("Kubernetes", matched)
        self.assertIn("Rust", missing)


class TestScorer(unittest.TestCase):
    def test_tfidf_identical_texts(self):
        text = "Senior Python engineer building scalable machine learning microservices with Docker and SQL."
        score = compute_tfidf_similarity(text, text)
        self.assertGreaterEqual(score, 99.0)

    def test_tfidf_disjoint_texts(self):
        text_a = "Pastry chef baking artisanal French sourdough croissants and tarts."
        text_b = "Kubernetes infrastructure engineer administering Linux kernels and BGP routing."
        score = compute_tfidf_similarity(text_a, text_b)
        self.assertLess(score, 10.0)

    def test_skill_match_exact_required(self):
        resume_text = "Experienced in Python, SQL, and Docker."
        required = ["Python", "SQL", "Docker"]
        res = calculate_skill_match_score(resume_text, required_skills=required)
        self.assertEqual(res["skill_score"], 100.0)
        self.assertEqual(len(res["missing_required"]), 0)

    def test_skill_match_partial_required(self):
        resume_text = "Experienced in Python only."
        required = ["Python", "Docker", "Kubernetes", "AWS"]
        res = calculate_skill_match_score(resume_text, required_skills=required)
        # 1 out of 4 matched = 25.0%
        self.assertEqual(res["skill_score"], 25.0)
        self.assertEqual(res["matched_required"], ["Python"])
        self.assertEqual(len(res["missing_required"]), 3)

    def test_skill_match_fallback_when_no_required(self):
        jd_text = "Looking for a specialist with Python, SQL, and Docker skills."
        resume_text = "Experienced in Python and SQL."
        res = calculate_skill_match_score(
            resume_text=resume_text,
            required_skills=[],
            jd_text=jd_text,
        )
        self.assertTrue(res["fallback_used"])
        # Should detect Python, SQL, Docker in JD, match 2/3 = 66.7%
        self.assertGreater(res["skill_score"], 60.0)

    def test_overall_weighted_score_formula(self):
        resume_data = {
            "candidate_name": "Test Candidate",
            "filename": "test.pdf",
            "text": "Senior Python developer with SQL, Docker, and Git.",
        }
        jd_text = "We need a Senior Python developer with SQL, Docker, and Git."
        required = ["Python", "SQL", "Docker", "Git"]

        # If text is nearly identical (~100) and skills 4/4 match (100)
        # 70% * 100 + 30% * 100 = 100
        analysis = analyze_candidate_resume(
            resume_data=resume_data,
            job_description=jd_text,
            required_skills=required,
            threshold=70.0,
            weight_text=0.70,
            weight_skill=0.30,
        )
        self.assertGreaterEqual(analysis["overall_score"], 85.0)
        self.assertEqual(analysis["recommendation"], "Strong Match")

    def test_recommendation_brackets(self):
        rec_high = generate_recommendation_and_explanation(
            overall_score=85.0,
            threshold=70.0,
            matched_required=["Python"],
            missing_required=[],
            matched_preferred=[],
            missing_preferred=[],
            text_sim_score=80.0,
            skill_score=90.0,
        )
        self.assertEqual(rec_high["recommendation"], "Strong Match")

        rec_mid = generate_recommendation_and_explanation(
            overall_score=62.0,
            threshold=70.0,
            matched_required=["Python"],
            missing_required=["Docker"],
            matched_preferred=[],
            missing_preferred=[],
            text_sim_score=60.0,
            skill_score=64.0,
        )
        self.assertEqual(rec_mid["recommendation"], "Potential Match")

        rec_low = generate_recommendation_and_explanation(
            overall_score=35.0,
            threshold=70.0,
            matched_required=[],
            missing_required=["Python", "Docker"],
            matched_preferred=[],
            missing_preferred=[],
            text_sim_score=30.0,
            skill_score=40.0,
        )
        self.assertEqual(rec_low["recommendation"], "Low Match")


class TestExtractorHeuristics(unittest.TestCase):
    def test_sanitize_filename(self):
        fn = "John_Doe_Resume_2024.pdf"
        self.assertEqual(sanitize_filename_to_name(fn), "John Doe")

    def test_extract_candidate_name_heuristic(self):
        text = "Jane Doe\nSoftware Engineer\njane@example.com\nExperience in Python..."
        name = extract_candidate_name_heuristic(text, fallback_name="Fallback")
        self.assertEqual(name, "Jane Doe")

    def test_extract_text_from_pdf_bytes(self):
        try:
            from src.demo_data import create_pdf_bytes
            from src.extractor import extract_text_from_pdf
        except (ImportError, ModuleNotFoundError):
            from demo_data import create_pdf_bytes
            from extractor import extract_text_from_pdf

        sample_text = "John Doe\nSoftware Engineer\njohn.doe@example.com\nExperienced in Python and Docker."
        pdf_bytes = create_pdf_bytes(sample_text)
        result = extract_text_from_pdf(pdf_bytes, filename="John_Doe_Resume.pdf")
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["candidate_name"], "John Doe")
        self.assertEqual(result["email"], "john.doe@example.com")
        self.assertIn("Python", result["text"])

    def test_scanned_pdf_detection(self):
        try:
            from src.demo_data import create_scanned_empty_pdf_bytes
            from src.extractor import extract_text_from_pdf
        except (ImportError, ModuleNotFoundError):
            from demo_data import create_scanned_empty_pdf_bytes
            from extractor import extract_text_from_pdf

        scanned_bytes = create_scanned_empty_pdf_bytes()
        result = extract_text_from_pdf(scanned_bytes, filename="Empty_Scanned.pdf")
        self.assertFalse(result["is_valid"])
        self.assertTrue(result["is_scanned_or_empty"])
        self.assertIn("scanned image", result["error"].lower())

    def test_skill_alias_matching(self):
        resume_text = "Experienced in deep learning models using torch, with postgres db storage and rest apis."
        target_skills = ["PyTorch", "SQL", "FastAPI"]
        matched, missing = match_target_skills(resume_text, target_skills)
        self.assertIn("PyTorch", matched)
        self.assertIn("SQL", matched)
        self.assertIn("FastAPI", matched)

    def test_experience_and_education_fit(self):
        try:
            from src.scorer import extract_experience_and_education
        except (ImportError, ModuleNotFoundError):
            from scorer import extract_experience_and_education

        text = "Dr. Alice Smith\nPh.D. in Computer Science\nOver 6 years of experience in data science."
        res = extract_experience_and_education(text, target_years=5.0, target_edu="Bachelor")
        self.assertGreaterEqual(res["years_detected"], 5.0)
        self.assertEqual(res["edu_label"], "Ph.D. / Doctorate")
        self.assertEqual(res["experience_fit_score"], 100.0)

    def test_job_presets_structure(self):
        try:
            from src.demo_data import JOB_PRESETS
        except (ImportError, ModuleNotFoundError):
            from demo_data import JOB_PRESETS

        self.assertIn("Senior AI / Machine Learning Engineer", JOB_PRESETS)
        self.assertIn("Data Analyst / BI Specialist", JOB_PRESETS)
        self.assertIn("Backend Software Engineer", JOB_PRESETS)
        self.assertIn("Frontend UI/UX Engineer", JOB_PRESETS)
        self.assertIn("Full Stack Python Developer", JOB_PRESETS)
        self.assertIn("Cloud & DevOps Engineer", JOB_PRESETS)

        for name, preset in JOB_PRESETS.items():
            self.assertIn("required_skills", preset)
            self.assertIn("description_text", preset)
            self.assertIn("threshold", preset)


if __name__ == "__main__":
    unittest.main()
