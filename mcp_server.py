from mcp.server import MCPServer
import pandas as pd

# Create MCP server
mcp = MCPServer("CareerMatch AI")

# Load job database
jobs = pd.read_csv("Data/jobs.csv")


# Tool 1: Search jobs
@mcp.tool()
def search_jobs(query: str) -> str:
    """Search for jobs based on job title, skills, or category."""

    query = query.lower()

    results = jobs[
        jobs["job_title"].str.lower().str.contains(query)
        | jobs["skills"].str.lower().str.contains(query)
        | jobs["category"].str.lower().str.contains(query)
    ]

    if results.empty:
        return "No matching jobs found."

    output = []

    for _, job in results.iterrows():
        output.append(
            f"Job: {job['job_title']}\n"
            f"Skills: {job['skills']}\n"
            f"Category: {job['category']}\n"
        )

    return "\n".join(output)


# Tool 2: Get job details
@mcp.tool()
def get_job_details(job_title: str) -> str:
    """Get complete information about a specific job."""

    result = jobs[
        jobs["job_title"].str.lower() == job_title.lower()
    ]

    if result.empty:
        return "Job not found."

    job = result.iloc[0]

    return (
        f"Job: {job['job_title']}\n"
        f"Skills: {job['skills']}\n"
        f"Degree: {job['degree']}\n"
        f"Category: {job['category']}\n"
        f"Learning Path: {job['learning_path']}"
    )


# Tool 3: Analyze skill gap
@mcp.tool()
def analyze_skill_gap(job_title: str, user_skills: str) -> str:
    """Find the skills missing for a particular job."""

    result = jobs[
        jobs["job_title"].str.lower() == job_title.lower()
    ]

    if result.empty:
        return "Job not found."

    job = result.iloc[0]

    required_skills = set(
        skill.strip().lower()
        for skill in job["skills"].split(",")
    )

    student_skills = set(
        skill.strip().lower()
        for skill in user_skills.split(",")
    )

    missing = required_skills - student_skills

    if not missing:
        return "You have all the required skills for this job!"

    return (
        f"Job: {job['job_title']}\n"
        f"Missing Skills: {', '.join(sorted(missing))}"
    )


# Tool 4: Get learning path
@mcp.tool()
def get_learning_path(job_title: str) -> str:
    """Get the recommended learning path for a job."""

    result = jobs[
        jobs["job_title"].str.lower() == job_title.lower()
    ]

    if result.empty:
        return "Job not found."

    job = result.iloc[0]

    return (
        f"Learning Path for {job['job_title']}:\n"
        f"{job['learning_path']}"
    )

# Tool 5: Recommend jobs
@mcp.tool()
def recommend_jobs(user_skills: str) -> str:
    """Recommend jobs based on the user's skills."""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    user_text = user_skills.lower().replace(",", " ")

    job_text = (
        jobs["skills"]
        .str.lower()
        .str.replace(",", " ", regex=False)
    )

    vectorizer = TfidfVectorizer()

    all_text = pd.concat(
        [pd.Series([user_text]), job_text],
        ignore_index=True
    )

    matrix = vectorizer.fit_transform(all_text)

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    )

    results = jobs.copy()
    results["match"] = similarity[0] * 100

    results = results.sort_values(
        by="match",
        ascending=False
    )

    output = []

    for _, job in results.head(5).iterrows():

        output.append(
            f"Job: {job['job_title']}\n"
            f"Match: {job['match']:.2f}%\n"
            f"Category: {job['category']}\n"
            f"Skills: {job['skills']}\n"
        )

    return "\n".join(output)
# Start MCP server
if __name__ == "__main__":
    mcp.run()