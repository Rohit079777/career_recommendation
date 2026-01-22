# ===================== IMPORTS =====================
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Career Guide AI",
    layout="wide",
    page_icon="🎯"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
.card {
    background: white;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    text-align: center;
    transition: all 0.3s ease-in-out;
    cursor: pointer;
}
.card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 15px 35px rgba(0,0,0,0.2);
}
.score {
    font-size: 34px;
    font-weight: bold;
}
.learn-box {
    background: #fff8dc;
    padding: 10px;
    border-radius: 10px;
    margin-top: 10px;
}
.badge {
    font-weight: 600;
}
.learn-button {
    background-color: #ff5722;
    color: white;
    padding: 6px 12px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    margin-left: 5px;
}
.learn-button:hover {
    background-color: #e64a19;
}
</style>
""", unsafe_allow_html=True)

# ===================== DATA =====================
data = {
    "Career": [
        "Data Scientist","Web Developer","AI Engineer","UI/UX Designer",
        "Machine Learning Engineer","Backend Developer","Frontend Developer"
    ],
    "Required_Skills": [
        "Python, Machine Learning, Statistics, SQL",
        "HTML, CSS, JavaScript, React, Git",
        "Python, Deep Learning, TensorFlow, NLP",
        "Adobe XD, Figma, UX Research, Graphic Design",
        "Python, Machine Learning, TensorFlow",
        "Python, Java, Node.js, SQL, APIs, Git",
        "HTML, CSS, JavaScript, React"
    ],
    "Image": [
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        "https://cdn-icons-png.flaticon.com/512/2721/2721297.png",
        "https://cdn-icons-png.flaticon.com/512/4712/4712035.png",
        "https://cdn-icons-png.flaticon.com/512/4207/4207253.png",
        "https://cdn-icons-png.flaticon.com/512/2103/2103832.png",
        "https://cdn-icons-png.flaticon.com/512/6213/6213731.png",
        "https://cdn-icons-png.flaticon.com/512/1055/1055687.png"
    ],
    "Description": [
        "Analyze data and build predictive ML models.",
        "Develop responsive and interactive websites.",
        "Build intelligent AI systems using DL & NLP.",
        "Design user-friendly app and web interfaces.",
        "Develop and optimize ML pipelines.",
        "Build scalable backend systems & APIs.",
        "Create frontend UI using modern frameworks."
    ],
    "Learn_Link": [
        "https://www.youtube.com/playlist?list=PLZoTAELRMXVPBTrgt0k5ax-pAw3F8q6Aa",  # Data Science
        "https://www.youtube.com/playlist?list=PLgUwDviBIf0qUlt5H_kiKYaNSqJ81PMMY",  # Web Dev
        "https://www.youtube.com/playlist?list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S",  # AI Engineer
        "https://www.youtube.com/playlist?list=PLF4FvftkVkb3nOjcslz5h1_8I6Tb9HtLx",  # UI/UX
        "https://www.youtube.com/playlist?list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S",  # ML Engineer
        "https://www.youtube.com/playlist?list=PLjwdMgw5TTLVDv-ceONHM_C19nYc6V-2y",  # Backend
        "https://www.youtube.com/playlist?list=PLgUwDviBIf0qUlt5H_kiKYaNSqJ81PMMY"   # Frontend
    ]
}

df = pd.DataFrame(data)

# ===================== FUNCTIONS =====================
def similarity(user, skills):
    vec = TfidfVectorizer()
    m = vec.fit_transform(skills)
    u = vec.transform([user])
    return cosine_similarity(u, m).flatten()

def badge(score):
    if score >= 80:
        return "🏆 Excellent Fit"
    elif score >= 60:
        return "🔥 Good Fit"
    else:
        return "⚠️ Needs Improvement"

def missing(user_set, req):
    return set(s.strip().lower() for s in req.split(",")) - user_set

def radar_chart(row, user_set):
    req = set(s.strip().lower() for s in row["Required_Skills"].split(","))
    skills = list(req | user_set)
    df_radar = pd.DataFrame({
        "Skill": skills * 2,
        "Value": [1 if s in user_set else 0 for s in skills] +
                 [1 if s in req else 0 for s in skills],
        "Type": ["You"]*len(skills) + ["Required"]*len(skills)
    })
    fig = px.line_polar(
        df_radar,
        r="Value",
        theta="Skill",
        color="Type",
        line_close=True,
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, use_container_width=True)

# ===================== UI =====================
st.title("🎯 Career Guide AI")
st.write("AI-powered career recommendation with skill gap analysis")

user_input = st.text_area(
    "🧠 Enter your skills (comma separated)",
    "Python, SQL, HTML"
)

user_skills = set(s.strip().lower() for s in user_input.split(","))

if st.button("🚀 Analyze My Career"):
    df["Match_Score"] = similarity(user_input, df["Required_Skills"]) * 100
    df = df.sort_values("Match_Score", ascending=False).reset_index(drop=True)

    # ===================== TOP 3 CARDS =====================
    st.markdown("## 🏆 Top 3 Matches")

    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]

    for col, (_, row) in zip(cols, df.head(3).iterrows()):
        with col:
            miss = missing(user_skills, row["Required_Skills"])
            learn_button_html = ''
            if miss:
                for skill in list(miss)[:3]:
                    learn_button_html += f'<a href="{row["Learn_Link"]}" target="_blank"><button class="learn-button">Learn {skill}</button></a>'
            st.markdown(f"""
            <div class="card">
                <img src="{row['Image']}" width="90"/>
                <h3>{row['Career']}</h3>
                <div class="badge">{badge(row['Match_Score'])}</div>
                <div class="score">{row['Match_Score']:.1f}%</div>
                <p>{row['Description']}</p>
                <div class="learn-box">
                    {learn_button_html if miss else 'You are ready 🎉'}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ===================== ANALYTICS =====================
    st.markdown("---")
    colA, colB = st.columns(2)

    with colA:
        st.subheader("📊 Career Match Overview")
        fig = px.bar(
            df.head(7),
            x="Match_Score",
            y="Career",
            orientation="h",
            color="Match_Score",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.subheader("🎯 Deep Dive: Top Career")
        radar_chart(df.iloc[0], user_skills)

# ===================== FOOTER =====================
st.markdown("---")
st.caption("Built with ❤️ by Rohit | Career Guide AI v3.1")
