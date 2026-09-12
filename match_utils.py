"""
Shared job-matching logic for CareerMatch AI.

Both app.py (Streamlit UI) and mcp_server.py (MCP tools) import from
here, so the match percentage a user sees on the dashboard is always
identical to the one the MCP Assistant reports.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_skills(text):
    """Turn a comma-separated skills string into a clean lowercase set."""
    return set(s.strip().lower() for s in str(text).split(",") if s.strip())


def skill_overlap_score(user_skills, job_skills_text):
    """
    Percentage of a job's REQUIRED skills the user actually has.
    This is the number people intuitively read as "match %" and is
    what should drive the missing-skills list.
    """
    required = normalize_skills(job_skills_text)
    if not required:
        return 0.0
    have = required & user_skills
    return (len(have) / len(required)) * 100


def tfidf_similarity_scores(user_text, job_text_series):
    """
    TF-IDF cosine similarity as a secondary signal, so jobs with
    closely related (but not identically spelled) skills still rank
    reasonably rather than scoring zero.
    """
    vectorizer = TfidfVectorizer()
    all_text = pd.concat([pd.Series([user_text]), job_text_series], ignore_index=True)
    matrix = vectorizer.fit_transform(all_text)
    similarity = cosine_similarity(matrix[0:1], matrix[1:])
    return similarity[0] * 100


def get_matches(jobs_df, skills_text, overlap_weight=0.7, tfidf_weight=0.3):
    """
    Return a copy of jobs_df sorted by a blended match score:
      - overlap_weight: weight given to exact required-skill overlap
      - tfidf_weight: weight given to general text similarity

    Defaults favor overlap (0.7) because it's the number that agrees
    with the "missing skills" list shown in the UI.
    """
    if not str(skills_text).strip():
        results = jobs_df.copy()
        results["match"] = 0.0
        return results

    user_skills = normalize_skills(skills_text)
    user_text = str(skills_text).lower().replace(",", " ")
    job_text = jobs_df["skills"].str.lower().str.replace(",", " ", regex=False)

    tfidf_scores = tfidf_similarity_scores(user_text, job_text)
    overlap_scores = jobs_df["skills"].apply(lambda s: skill_overlap_score(user_skills, s))

    results = jobs_df.copy()
    results["match"] = (overlap_weight * overlap_scores) + (tfidf_weight * tfidf_scores)
    results["match"] = results["match"].clip(0, 100)
    return results.sort_values("match", ascending=False).reset_index(drop=True)