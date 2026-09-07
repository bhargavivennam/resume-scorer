"""Unit tests for feature extraction — the layer everything else depends on."""
import pytest

from features import (Interval, experience_and_gaps, extract_features,
                      merge_intervals, parse_date_ranges)


def test_parses_month_year_ranges():
    ivals = parse_date_ranges("jan 2019 - mar 2021")
    assert len(ivals) == 1
    assert ivals[0].end - ivals[0].start == 26  # 26 months


def test_parses_present_and_bare_years():
    assert len(parse_date_ranges("2018 - present")) == 1
    assert len(parse_date_ranges("2015 – 2017")) == 1


def test_merge_removes_double_counting():
    # Two concurrent roles must not count as 4 years of experience.
    merged = merge_intervals([Interval(0, 24), Interval(12, 36)])
    assert merged == [Interval(0, 36)]


def test_gap_detection():
    text = "jan 2015 - jan 2017 ... jan 2019 - jan 2021"
    years, n_gaps, longest = experience_and_gaps(text)
    assert n_gaps == 1
    assert longest == 24
    assert years == pytest.approx(4.0)


def test_short_break_is_not_a_gap():
    _, n_gaps, _ = experience_and_gaps("jan 2015 - jan 2017 ... apr 2017 - jan 2019")
    assert n_gaps == 0  # 3 months < the 6-month threshold


def test_skill_extraction_handles_punctuated_names():
    feats = extract_features("Skills: Python, C++, Node.js, PostgreSQL, AWS, PyTorch")
    assert feats["n_skills"] >= 5
    assert feats["skills_cloud_devops"] >= 1
    assert feats["skills_ml"] >= 1


def test_does_not_match_skill_substrings():
    # "rust" must not fire on "trusted", "go" must not fire on "google".
    feats = extract_features("A trusted engineer who used google docs")
    assert feats["skills_languages"] == 0


def test_degrees_and_certifications():
    feats = extract_features("M.S. Computer Science\nB.Tech\nAWS Certified Solutions Architect")
    assert feats["n_degrees"] >= 2
    assert feats["n_certifications"] >= 1
    assert feats["has_advanced_degree"] == 1


def test_quantified_and_weak_language():
    strong = extract_features("- Reduced p95 latency by 40% and saved $30k")
    weak = extract_features("- Responsible for code. Worked on things. Helped with tests.")
    assert strong["n_quantified_results"] >= 2
    assert weak["n_weak_phrases"] >= 3
    assert weak["n_quantified_results"] == 0


def test_contact_detection():
    feats = extract_features("a@b.com | +1 (555) 123-4567 | github.com/x")
    assert feats["has_email"] == feats["has_phone"] == feats["has_links"] == 1


def test_empty_input_is_safe():
    feats = extract_features("")
    assert feats["years_experience"] == 0
    assert feats["word_count"] == 0


def test_feature_vector_is_stable():
    """Every resume must produce the same keys in the same order, or the
    model's coefficients would silently line up with the wrong features."""
    a = list(extract_features("hello").keys())
    b = list(extract_features("Python engineer, 2019 - 2024").keys())
    assert a == b


def test_education_dates_do_not_count_as_experience():
    """Degree date ranges must not inflate years of experience."""
    resume = (
        "EXPERIENCE\n"
        "Engineer - Acme\n"
        "Jan 2020 - Jan 2024\n"
        "- Built things\n\n"
        "EDUCATION\n"
        "B.Tech Computer Science, State University, 2012 - 2016\n"
    )
    feats = extract_features(resume)
    assert feats["years_experience"] == pytest.approx(4.0)
    assert feats["n_roles"] == 1
    assert feats["n_employment_gaps"] == 0  # the 2016->2020 hole is not a real gap


def test_falls_back_to_whole_text_without_an_experience_header():
    feats = extract_features("Engineer at Acme, Jan 2020 - Jan 2024")
    assert feats["years_experience"] == pytest.approx(4.0)


def test_redaction_removes_identity_and_years():
    from features import redact_for_text_model
    out = redact_for_text_model(
        "Alex Chen\na@b.com | +1 (555) 201-3344 | linkedin.com/in/alex\n"
        "Engineer, Mar 2019 - Present\n- Cut latency 40% with Kubernetes")
    assert "Alex Chen" not in out      # first line (the name) is dropped
    assert "a@b.com" not in out
    assert "linkedin" not in out
    assert "2019" not in out           # graduation-year / age proxy
    assert "Kubernetes" in out         # substance survives
    assert "40%" in out


def test_role_parsing_attaches_dates_to_bullets():
    from features import parse_roles
    roles = parse_roles(
        "EXPERIENCE\nStaff Engineer\nJan 2020 - Present\n- did a thing\n"
        "Engineer\nJan 2014 - Dec 2016\n- did another")
    assert len(roles) == 2
    assert roles[0][1] and roles[1][1] and roles[0][1] > roles[1][1]


def test_evidence_features_reward_demonstration():
    from features import extract_features
    listed = extract_features(
        "EXPERIENCE\nEngineer\nJan 2020 - Present\n- did stuff\nSKILLS\nKubernetes, Python")
    shown = extract_features(
        "EXPERIENCE\nEngineer\nJan 2020 - Present\n"
        "- Cut deploy time 55% migrating to Kubernetes\n- Built Python services\n"
        "SKILLS\nKubernetes, Python")
    assert shown["n_demonstrated_skills"] > listed["n_demonstrated_skills"]
    assert shown["skill_strength_mean"] > listed["skill_strength_mean"]


def test_structural_features_are_not_label_proxies():
    """Guard against shortcut learning at the SOURCE — the generated data.

    If a purely structural feature (length, bullet count, seniority) correlates
    strongly with the grade, the model will learn that shortcut instead of
    learning quality. This test fails loudly if a future change to the
    generator reintroduces one.
    """
    from scipy.stats import spearmanr

    from data import generate_graded_dataset
    from features import extract_features

    texts, grades = generate_graded_dataset(400, seed=101)
    feats = [extract_features(t) for t in texts]

    # Structure must be weakly informative at most.
    for name, limit in (("word_count", 0.30), ("n_bullets", 0.30),
                        ("years_experience", 0.25), ("n_certifications", 0.30),
                        ("n_sections", 0.30)):
        rho = abs(spearmanr([f[name] for f in feats], grades).statistic)
        assert rho < limit, f"{name} correlates {rho:.2f} with the label — a shortcut"

    # Content must be the strongest signal available.
    content = max(
        spearmanr([f[n] for f in feats], grades).statistic
        for n in ("n_quantified_results", "n_demonstrated_skills"))
    structure = max(
        abs(spearmanr([f[n] for f in feats], grades).statistic)
        for n in ("word_count", "n_bullets", "n_sections"))
    assert content > structure, "structure outweighs content in the training data"
