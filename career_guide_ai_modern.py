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
    justify-content: flex-start;
}
.badge {
    font-weight: 600;
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

# ===================== LOAD CSV DATA =====================
df = pd.read_csv("career_dataset_100.csv",encoding="latin1")
df["Salary"] = (
    df["Salary"]
    .astype(str)
    .str.replace("–", "-", regex=False)
)

# Rupees symbol forcefully add
df["Salary"] = "₹" + df["Salary"]
df.columns = df.columns.str.strip()

# ===================== FIXED MATCH LOGIC =====================
def match_score(user_skills, required_skills):
    user = set(s.lower() for s in user_skills)
    req = set(s.strip().lower() for s in required_skills.split(","))
    return (len(user & req) / len(req)) * 100 if req else 0

# ===================== OTHER FUNCTIONS (UNCHANGED) =====================
def badge(score):
    if score >= 80:
        return "🏆 Excellent Fit"
    elif score >= 60:
        return "🔥 Good Fit"
    else:
        return "⚠️ Needs Improvement"

def missing(user_set, req):
    user_lower = set(s.lower() for s in user_set)
    req_lower = set(s.strip().lower() for s in req.split(","))
    return req_lower - user_lower

def radar_chart(row, user_set):
    req = set(s.strip().lower() for s in row["Required_Skills"].split(","))
    skills = list(req | set(s.lower() for s in user_set))
    df_radar = pd.DataFrame({
        "Skill": skills * 2,
        "Value": [1 if s in set(x.lower() for x in user_set) else 0 for s in skills] +
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

all_skills = sorted({skill.strip() for req in df['Required_Skills'] for skill in req.split(',')})
user_skills = set(st.multiselect("🧠 Select your skills", all_skills, default=["Python","SQL","HTML"]))

if st.button("🚀 Analyze My Career"):

    if not user_skills:
        st.warning("⚠️ Please select at least one skill!")
    else:
        # ✅ FIXED SCORE (NO TF-IDF)
        df["Match_Score"] = df["Required_Skills"].apply(
            lambda x: match_score(user_skills, x)
        )

        df = df.sort_values("Match_Score", ascending=False).reset_index(drop=True)

        # ===================== TOP 3 CARDS =====================
        st.markdown("## 🏆 Top 3 Matches")
        c1, c2, c3 = st.columns(3)

        for col, (_, row) in zip([c1, c2, c3], df.head(3).iterrows()):
            with col:
                miss = missing(user_skills, row["Required_Skills"])
                miss_text = " • ".join(skill.title() for skill in miss) if miss else ""

                # salary html (clean)
                salary_html = f"""
                <div style="background-color:#4caf50;color:white;
                padding:6px 12px;border-radius:8px;font-weight:bold;">
                💰 Estimated Salary: {row["Salary"]} per annum
                </div>
                """

                learn_button_html = (
                    f'<a href="{row["Learn_Link"]}" target="_blank" '
                    f'style="background-color:#ff5722;color:white;'
                    f'padding:6px 12px;border-radius:8px;'
                    f'text-decoration:none;display:inline-block;font-weight:bold;">'
                    f'Learn Missing Skills</a>'
                ) if miss else salary_html

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
                    <div class="learn-box">
                        {"Missed Skills: • " + miss_text if miss else "You are ready 🎉"}
                    </div>
                    <div style="margin-top:10px;">{learn_button_html}</div>
                </div>
                """, unsafe_allow_html=True)

        # ===================== ANALYTICS =====================
        st.markdown("---")

        st.markdown("### 📊 Career Match Overview")
        top7 = df.head(7)
        fig_bar = px.bar(
            top7,
            x="Match_Score",
            y="Career",
            orientation="h",
            color="Match_Score",
            color_continuous_scale=px.colors.sequential.Plasma,
            text=top7["Match_Score"].apply(lambda x: f"{x:.2f}%")
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 🎯 Deep Dive: Top Career")
        radar_chart(df.iloc[0], user_skills)

        st.markdown("### 🥇🥈🥉 Top 3 Career Match Distribution")
        top3 = df.head(3).sort_values("Match_Score", ascending=False)
        fig_pie = px.pie(
            top3,
            names='Career',
            values='Match_Score',
            color='Career',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.4
        )
        fig_pie.update_traces(
            texttemplate=top3["Match_Score"].apply(lambda x: f"{x:.2f}%"),
            textposition='inside'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### 📈 Top 10 Career Match Trend")
        top10 = df.head(10)
        fig_line = px.line(
            top10,
            x='Career',
            y='Match_Score',
            markers=True,
            text=top10['Match_Score'].apply(lambda x: f"{x:.2f}%"),
            color_discrete_sequence=['#ff5722']
        )
        fig_line.update_traces(marker=dict(size=10))
        st.plotly_chart(fig_line, use_container_width=True)

# ===================== FOOTER =====================
st.markdown("---")
st.caption("Built with ❤️ by Rohit | Career Guide AI v5.5")
