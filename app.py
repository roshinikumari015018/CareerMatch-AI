import streamlit as st
import textwrap
import pandas as pd
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
from mcp_client import recommend_jobs

st.set_page_config(
    page_title="CareerMatch AI",
    page_icon="🎓",
    layout="wide"
)

# ---------- GLOBAL STYLE ----------
st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}

/* Sidebar dark theme */
section[data-testid="stSidebar"] {
    background-color: #1e2749;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

h1, h2, h3 {
    color: #1e293b !important;
}

/* Metric card */
.metric-card {
    border-radius: 16px;
    padding: 20px;
    height: 100%;
}
.metric-label {
    font-size: 14px;
    font-weight: 600;
    opacity: 0.85;
}
.metric-value {
    font-size: 30px;
    font-weight: 800;
    margin: 4px 0;
}
.metric-sub {
    font-size: 12px;
    opacity: 0.7;
}

/* Job row card */
.job-row {
    background: white;
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
    border: 1px solid #eef0f4;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.job-icon {
    font-size: 26px;
    background: #f1f5f9;
    border-radius: 10px;
    width: 46px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 14px;
}
.job-title {
    font-weight: 700;
    font-size: 15px;
    color: #0f172a;
}
.job-meta {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 6px;
}
.tag {
    background: #f1f5f9;
    color: #475569;
    padding: 2px 9px;
    border-radius: 8px;
    font-size: 11px;
    margin-right: 5px;
}
.missing-pill {
    background: #fee2e2;
    color: #dc2626;
    padding: 4px 10px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
    margin-top: 4px;
}
.section-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #eef0f4;
}
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    color: white;
    font-size: 12px;
    font-weight: 700;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------- DATA ----------
jobs = pd.read_csv("Data/jobs.csv")

CATEGORY_ICONS = {
    "Software": "💻",
    "Data": "📊",
    "AI": "🤖",
    "Web Development": "🌐",
    "Cybersecurity": "🛡️",
    "Cloud": "☁️"
}

STEP_COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"]


def ring_color(pct):
    if pct >= 70:
        return "#22c55e"
    elif pct >= 50:
        return "#3b82f6"
    else:
        return "#f97316"

def render_ring(pct, size=70):
    color = ring_color(pct)
    inner = size - 16
    html = f"""
    <div style="width:{size}px;height:{size}px;border-radius:50%;
        background:conic-gradient({color} {pct}%, #e5e7eb {pct}% 100%);
        display:flex;align-items:center;justify-content:center;">
        <div style="width:{inner}px;height:{inner}px;border-radius:50%;background:white;
            display:flex;flex-direction:column;align-items:center;justify-content:center;">
            <span style="font-weight:800;font-size:15px;color:#0f172a;">{pct:.0f}%</span>
            <span style="font-size:8px;color:#94a3b8;">Match</span>
        </div>
    </div>
    """
    return textwrap.dedent(html).strip()



def render_metric_card(icon, label, value, sub, bg, accent):
    html = f"""
    <div class="metric-card" style="background:{bg};">
        <div style="font-size:22px;">{icon}</div>
        <div class="metric-label" style="color:{accent};">{label}</div>
        <div class="metric-value" style="color:{accent};">{value}</div>
        <div class="metric-sub" style="color:{accent};">{sub}</div>
    </div>
    """
    return textwrap.dedent(html).strip()


def get_matches(skills_text):
    """Return jobs dataframe sorted by match % for given comma skills text."""
    user_text = skills_text.lower().replace(",", " ")
    job_text = jobs["skills"].str.lower().str.replace(",", " ", regex=False)
    vectorizer = TfidfVectorizer()
    all_text = pd.concat([pd.Series([user_text]), job_text], ignore_index=True)
    matrix = vectorizer.fit_transform(all_text)
    similarity = cosine_similarity(matrix[0:1], matrix[1:])
    results = jobs.copy()
    results["match"] = similarity[0] * 100
    return results.sort_values("match", ascending=False)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 🎓 CareerMatch AI")
    st.caption("Your Skills • Better Jobs • Brighter Future")

    selected = option_menu(
        menu_title=None,
        options=[
            "Dashboard",
            "Job Recommendations",
            "Skill Gap Analysis",
            "Learning Path",
            "MCP Assistant"
        ],
        icons=["house", "briefcase", "bullseye", "book", "robot"],
        default_index=0,
        styles={
            "container": {"background-color": "#1e2749"},
            "icon": {"color": "#93c5fd", "font-size": "16px"},
            "nav-link": {
                "color": "#e2e8f0",
                "font-size": "14px",
                "text-align": "left",
                "margin": "4px 0",
                "border-radius": "8px"
            },
            "nav-link-selected": {"background-color": "#3b82f6"},
        }
    )

st.session_state.page = selected


# ---------- HEADER ----------
name = st.session_state.get("student_name", "")
greeting_name = name if name else "there"

st.title(f"Hello, {greeting_name} 👋")
st.caption("Let's build your dream career together with CareerMatch AI")
st.divider()


# ======================================================
# PAGE: DASHBOARD
# ======================================================
if st.session_state.page == "Dashboard":

    skills_for_dashboard = st.session_state.get("resume_skills", "")

    if not skills_for_dashboard.strip():
        st.info("Add your skills on the **Job Recommendations** page to personalize your dashboard.")
        matches = jobs.copy()
        matches["match"] = 0
    else:
        matches = get_matches(skills_for_dashboard)

    top_matches = matches.head(4)
    best_match = matches.iloc[0]["match"] if len(matches) else 0
    all_missing = set()
    for _, job in top_matches.iterrows():
        req = set(s.strip().lower() for s in job["skills"].split(","))
        have = set(s.strip().lower() for s in skills_for_dashboard.split(",") if s.strip())
        all_missing |= (req - have)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_metric_card("💼", "Top Job Matches", f"{len(top_matches)}+",
                                        "Personalized recommendations", "#eaf1ff", "#2563eb"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_metric_card("🎯", "Highest Match", f"{best_match:.0f}%",
                                        "Best matched job", "#eafaf1", "#16a34a"), unsafe_allow_html=True)
    with c3:
        st.markdown(render_metric_card("📖", "Skills to Learn", f"{len(all_missing)}",
                                        "Improve your profile", "#f4eeff", "#7c3aed"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_metric_card("📈", "Jobs in Database", f"{len(jobs)}",
                                        "Career paths tracked", "#ffeef0", "#e11d48"), unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("💼 Top Job Recommendations")

        for _, job in top_matches.iterrows():
            icon = CATEGORY_ICONS.get(job["category"], "📌")
            req_skills = [s.strip() for s in job["skills"].split(",")]
            have = set(s.strip().lower() for s in skills_for_dashboard.split(",") if s.strip())
            missing = [s for s in req_skills if s.strip().lower() not in have]

            col_a, col_b, col_c = st.columns([3, 1, 2])
            with col_a:
                st.markdown(f"""
                <div style="display:flex;align-items:center;">
                    <div class="job-icon">{icon}</div>
                    <div>
                        <div class="job-title">{job['job_title']}</div>
                        <div class="job-meta">{job['category']} • {job['degree']}</div>
                        <div>{''.join(f'<span class="tag">{s}</span>' for s in req_skills)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(render_ring(job["match"]), unsafe_allow_html=True)
            with col_c:
                if missing:
                    st.markdown(f'<div class="missing-pill">Missing: {", ".join(missing)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="missing-pill" style="background:#dcfce7;color:#16a34a;">All skills matched!</div>', unsafe_allow_html=True)
            st.write("")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("🎯 Skill Gap Snapshot")

        if len(top_matches):
            top_job = top_matches.iloc[0]
            req = set(s.strip().lower() for s in top_job["skills"].split(","))
            have = set(s.strip().lower() for s in skills_for_dashboard.split(",") if s.strip())
            matched = req & have
            pct = (len(matched) / len(req) * 100) if req else 0

            st.markdown(f"""
            <div style="display:flex;justify-content:center;margin:10px 0;">
                {render_ring(pct, size=140)}
            </div>
            <p style="text-align:center;color:#64748b;font-size:13px;">
                Based on top match: <b>{top_job['job_title']}</b>
            </p>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("📘 Learning Path Preview")

        if len(top_matches):
            steps = [s.strip() for s in top_matches.iloc[0]["learning_path"].split(",") if s.strip()]
            for i, step in enumerate(steps, start=1):
                color = STEP_COLORS[(i - 1) % len(STEP_COLORS)]
                st.markdown(f"""
                <div style="display:flex;align-items:center;margin-bottom:10px;">
                    <div class="step-badge" style="background:{color};">{i}</div>
                    <span style="font-size:13px;color:#334155;">{step}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# PAGE: JOB RECOMMENDATIONS
# ======================================================
elif st.session_state.page == "Job Recommendations":

    st.header("Resume Analysis")
    st.write("Upload your resume and CareerMatch AI will extract your technical skills automatically.")

    uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

    if uploaded_resume is not None:
        try:
            reader = PdfReader(uploaded_resume)
            resume_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    resume_text += text + " "
            resume_text = resume_text.lower()

            known_skills = [
                "python", "java", "c", "c++", "sql", "html", "css", "javascript",
                "react", "git", "github", "linux", "aws", "docker",
                "machine learning", "pandas", "numpy", "scikit-learn",
                "spring boot", "networking", "security", "excel", "power bi"
            ]
            extracted_skills = [s for s in known_skills if s in resume_text]

            if extracted_skills:
                st.success("Resume analyzed successfully!")
                st.subheader("Extracted Skills")
                st.write(", ".join(s.title() for s in extracted_skills))
                st.session_state["resume_skills"] = ", ".join(extracted_skills)
            else:
                st.warning("No known technical skills were detected.")
        except Exception as e:
            st.error(f"Could not read the resume: {e}")

    st.header("Student Profile")
    col1, col2 = st.columns(2)

    with col1:
        name_input = st.text_input("Your Name", placeholder="Enter your name")
        if name_input:
            st.session_state["student_name"] = name_input

        degree = st.selectbox("Your Degree", [
            "BSc Computer Science", "BCA", "BTech Computer Science", "Other"
        ])

    with col2:
        default_skills = st.session_state.get("resume_skills", "")
        skills_input = st.text_input("Your Skills", value=default_skills, placeholder="Example: Java, SQL, Git")
        category = st.selectbox("Preferred Career Area", [
            "All", "Software", "Data", "AI", "Web Development", "Cybersecurity", "Cloud"
        ])

    if st.button("Find My Best Jobs", type="primary"):
        if not skills_input.strip():
            st.warning("Please enter at least one skill.")
        else:
            st.session_state["resume_skills"] = skills_input
            results = get_matches(skills_input)

            if category != "All":
                results = results[results["category"] == category]

            student_skills = set(x.strip().lower() for x in skills_input.split(",") if x.strip())

            st.success(f"Here are your personalized career matches, {name_input or 'there'}!")
            st.header("Top Job Recommendations")

            for number, (_, job) in enumerate(results.head(5).iterrows(), start=1):
                required_skills = set(x.strip().lower() for x in job["skills"].split(","))
                missing_skills = required_skills - student_skills

                st.markdown('<div class="job-row" style="display:block;">', unsafe_allow_html=True)
                st.subheader(f"{number}. {job['job_title']}")
                st.markdown(render_ring(job["match"]), unsafe_allow_html=True)
                st.write(f"**Category:** {job['category']}")
                st.write(f"**Required Skills:** {job['skills']}")

                if missing_skills:
                    st.write("**Missing Skills:** " + ", ".join(sorted(missing_skills)))
                else:
                    st.success("You have all required skills!")

                st.write("**Learning Path:**")
                for step_number, step in enumerate(
                    [s.strip() for s in job["learning_path"].split(",") if s.strip()], start=1
                ):
                    st.write(f"**Step {step_number}:** {step}")
                st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# PAGE: SKILL GAP ANALYSIS
# ======================================================
elif st.session_state.page == "Skill Gap Analysis":

    st.header("Skill Gap Analysis")
    st.write("Select a career to see your current skill match, missing skills, and recommended learning path.")

    gap_col1, gap_col2 = st.columns(2)
    with gap_col1:
        gap_job = st.selectbox("Select a Career", jobs["job_title"].tolist(), key="gap_job")
    with gap_col2:
        gap_skills = st.text_input("Enter Your Skills", placeholder="Example: Java, SQL, Git", key="gap_skills")

    if st.button("Analyze Skill Gap", type="primary"):
        if not gap_skills.strip():
            st.warning("Please enter at least one skill.")
        else:
            selected_job = jobs[jobs["job_title"] == gap_job].iloc[0]
            user_skill_set = set(s.strip().lower() for s in gap_skills.split(",") if s.strip())
            required_skill_set = set(s.strip().lower() for s in selected_job["skills"].split(",") if s.strip())

            matched_skills = user_skill_set & required_skill_set
            missing_skills = required_skill_set - user_skill_set
            match_percentage = len(matched_skills) / len(required_skill_set) * 100

            st.subheader(f"Analysis for {gap_job}")
            st.markdown(render_ring(match_percentage, size=120), unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Skill Match", f"{match_percentage:.0f}%")
            with c2:
                st.metric("Skills You Have", len(matched_skills))
            with c3:
                st.metric("Skills Missing", len(missing_skills))

            st.subheader("Your Skills")
            st.success(", ".join(sorted(matched_skills)) if matched_skills else "No required skills matched yet.")

            st.subheader("Required Skills")
            st.write(", ".join(sorted(required_skill_set)))

            st.subheader("Missing Skills")
            if missing_skills:
                st.warning(", ".join(sorted(missing_skills)))
            else:
                st.success("You have all the required skills for this career!")

            st.subheader("Recommended Learning Path")
            st.write(selected_job["learning_path"])


# ======================================================
# PAGE: LEARNING PATH
# ======================================================
elif st.session_state.page == "Learning Path":

    st.header("📘 Learning Path Lookup")
    lp_job = st.selectbox("Select a Career", jobs["job_title"].tolist(), key="lp_job")

    if st.button("Show Learning Path", type="primary"):
        selected = jobs[jobs["job_title"] == lp_job].iloc[0]
        st.subheader(f"Learning Path for {lp_job}")
        steps = [s.strip() for s in selected["learning_path"].split(",") if s.strip()]
        for i, step in enumerate(steps, start=1):
            color = STEP_COLORS[(i - 1) % len(STEP_COLORS)]
            st.markdown(f"""
            <div style="display:flex;align-items:center;margin-bottom:10px;">
                <div class="step-badge" style="background:{color};">{i}</div>
                <span style="font-size:14px;color:#334155;">{step}</span>
            </div>
            """, unsafe_allow_html=True)


# ======================================================
# PAGE: MCP ASSISTANT
# ======================================================
elif st.session_state.page == "MCP Assistant":

    st.header("CareerMatch AI MCP Assistant")
    st.write("Use MCP to get intelligent job recommendations from the CareerMatch AI job database.")

    mcp_skills = st.text_input("Enter skills for MCP", placeholder="Example: Java, SQL, Git")

    if st.button("🤖 Ask CareerMatch MCP", type="primary"):
        if not mcp_skills.strip():
            st.warning("Please enter your skills.")
        else:
            with st.spinner("MCP is analyzing your skills..."):
                try:
                    result = recommend_jobs(mcp_skills)
                    st.success("MCP recommendation completed!")

                    if result.content:
                        text = result.content[0].text
                        st.header("MCP Job Recommendations")
                        blocks = text.strip().split("\n\n")

                        for block in blocks:
                            lines = block.split("\n")
                            if len(lines) >= 4:
                                job_name = lines[0].replace("Job: ", "")
                                match = lines[1].replace("Match: ", "")
                                job_category = lines[2].replace("Category: ", "")
                                required = lines[3].replace("Skills: ", "")

                                st.subheader(job_name)
                                st.metric("Match", match)
                                st.write(f"**Category:** {job_category}")
                                st.write(f"**Required Skills:** {required}")
                                st.divider()
                except Exception as e:
                    st.error(f"MCP connection error: {e}")