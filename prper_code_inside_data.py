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
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    justify-content: center;
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
}
.learn-button:hover {
    background-color: #e64a19;
}
.progress-bar {
    background-color: #ff5722;
    height: 15px;
    border-radius: 8px;
}
.progress-container {
    background-color: #e0e0e0;
    border-radius: 8px;
    height: 15px;
    margin-top: 5px;
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
        "https://www.youtube.com/playlist?list=PLZoTAELRMXVPBTrgt0k5ax-pAw3F8q6Aa",
        "https://www.youtube.com/playlist?list=PLgUwDviBIf0qUlt5H_kiKYaNSqJ81PMMY",
        "https://www.youtube.com/playlist?list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S",
        "https://www.youtube.com/playlist?list=PLF4FvftkVkb3nOjcslz5h1_8I6Tb9HtLx",
        "https://www.youtube.com/playlist?list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S",
        "https://www.youtube.com/playlist?list=PLjwdMgw5TTLVDv-ceONHM_C19nYc6V-2y",
        "https://www.youtube.com/playlist?list=PLgUwDviBIf0qUlt5H_kiKYaNSqJ81PMMY"
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
    fig.update_traces(fill="toself", marker_size=6)
    fig.update_layout(margin=dict(l=30,r=30,t=30,b=30))
    st.plotly_chart(fig, use_container_width=True)

# ===================== UI =====================
st.title("🎯 Career Guide AI")
st.write("AI-powered career recommendation with skill gap analysis")

# Multiselect dropdown for skills
all_skills = sorted({skill.strip() for req in df['Required_Skills'] for skill in req.split(',')})
user_skills = set(st.multiselect("🧠 Select your skills", all_skills, default=["Python","SQL","HTML"]))

if st.button("🚀 Analyze My Career"):

    if not user_skills:
        st.warning("⚠️ Please select at least one skill!")
    else:
        user_input_str = ', '.join(user_skills)
        df["Match_Score"] = similarity(user_input_str, df["Required_Skills"]) * 100
        df = df.sort_values("Match_Score", ascending=False).reset_index(drop=True)

        # ===================== TOP 3 CARDS =====================
        st.markdown("## 🏆 Top 3 Matches")
        c1, c2, c3 = st.columns(3)
        for col, (_, row) in zip([c1, c2, c3], df.head(3).iterrows()):
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
                    <div class="score">{row['Match_Score']:.2f}%</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width:{row['Match_Score']:.2f}%;"></div>
                    </div>
                    <p>{row['Description']}</p>
                    <div class="learn-box" style="flex-wrap: wrap; gap: 10px; justify-content: flex-start;">
                        {learn_button_html if miss else 'You are ready 🎉'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ===================== ANALYTICS =====================
        st.markdown("---")

        # -------- ORIGINAL BAR CHART --------
        st.markdown("### 📊 Career Match Overview")
        fig_bar = px.bar(
            df.head(7),
            x="Match_Score",
            y="Career",
            orientation="h",
            color="Match_Score",
            color_continuous_scale=px.colors.sequential.Plasma,
            text=df["Match_Score"].apply(lambda x: f"{x:.2f}%")
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

        # -------- ORIGINAL RADAR CHART --------
        st.markdown("### 🎯 Deep Dive: Top Career")
        radar_chart(df.iloc[0], user_skills)

        # -------- NEW PIE CHART --------
        st.markdown("### 🥧 Top 3 Career Match Distribution")
        top3 = df.head(3).sort_values("Match_Score", ascending=False)  # sorted descending
        fig_pie = px.pie(
            top3,
            names='Career',
            values='Match_Score',
            color='Career',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        fig_pie.update_traces(
            texttemplate='%{value:.2f}%',
            textposition='inside',
            hovertemplate='%{label}: %{value:.2f}%<extra></extra>'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # -------- NEW LINE CHART --------
        st.markdown("### 📈 All Career Match Trend")
        fig_line = px.line(
            df,
            x='Career',
            y='Match_Score',
            markers=True,
            text=df['Match_Score'].apply(lambda x: f"{x:.2f}%"),
            color_discrete_sequence=['#ff5722']
        )
        fig_line.update_traces(marker=dict(size=10))
        fig_line.update_layout(xaxis_title="Career", yaxis_title="Match Score (%)")
        st.plotly_chart(fig_line, use_container_width=True)

# ===================== FOOTER =====================
st.markdown("---")
st.caption("Built with ❤️ by Rohit | Career Guide AI v5.4")
