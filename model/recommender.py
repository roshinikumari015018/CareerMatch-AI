import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load job data
jobs = pd.read_csv("Data/jobs.csv")

# Get skills from the user
user_input = input("Enter your skills separated by commas: ")

# Convert user skills into one text string
user_skills = user_input.lower().replace(",", " ")

# Convert job skills into text
job_skill_text = jobs["skills"].str.lower().str.replace(",", " ", regex=False)

# Create TF-IDF model
vectorizer = TfidfVectorizer()

# Combine user skills and job skills
all_skills = pd.concat(
    [pd.Series([user_skills]), job_skill_text],
    ignore_index=True
)

# Convert skills into numerical vectors
tfidf_matrix = vectorizer.fit_transform(all_skills)

# Compare user skills with every job
similarities = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
)

# Convert similarity into percentage
jobs["match"] = similarities[0] * 100

# Sort jobs from highest match to lowest
jobs = jobs.sort_values(by="match", ascending=False)


# Display results
print("\n================================")
print("       CAREERMATCH AI")
print("================================")

print("\nRecommended Jobs:\n")

for _, job in jobs.head(5).iterrows():

    job_skills = set(
        skill.strip().lower()
        for skill in job["skills"].split(",")
    )

    missing_skills = job_skills - set(
        skill.strip().lower()
        for skill in user_input.split(",")
    )

    print(job["job_title"], "-", round(job["match"], 2), "% match")

    if missing_skills:
        print("  Missing skills:", ", ".join(missing_skills))
    else:
        print("  You have all the required skills!")

    print("  Learning Path:", job["learning_path"])
    print()