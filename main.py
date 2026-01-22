# ===================== IMPORTS =====================
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from io import BytesIO
import time

# ===================== DATA =====================
data = {
    "Career": [
        "Data Scientist","Web Developer","AI Engineer","UI/UX Designer",
        "Machine Learning Engineer","Backend Developer","Frontend Developer",
        "Business Analyst","Software Tester","Cloud Engineer","Cybersecurity Analyst"
    ],
    "Required_Skills": [
        "Python, Machine Learning, Statistics, SQL",
        "HTML, CSS, JavaScript, React, Git",
        "Python, Deep Learning, TensorFlow, NLP",
        "Adobe XD, Figma, UX Research, Graphic Design",
        "Python, Machine Learning, TensorFlow, Scikit-learn",
        "Python, Java, Node.js, SQL, APIs, Git",
        "HTML, CSS, JavaScript, React, UI Design",
        "Excel, SQL, Power BI, Business Analysis, Communication",
        "Manual Testing, Automation Testing, Selenium, Test Cases",
        "AWS, Azure, Docker, Kubernetes, Cloud Computing",
        "Network Security, Firewall, Ethical Hacking, Python, Security Analysis"
    ],
    "Image": [
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        "https://cdn-icons-png.flaticon.com/512/2721/2721297.png",
        "https://cdn-icons-png.flaticon.com/512/4712/4712035.png",
        "https://cdn-icons-png.flaticon.com/512/4207/4207253.png",
        "https://cdn-icons-png.flaticon.com/512/2103/2103832.png",
        "https://cdn-icons-png.flaticon.com/512/6213/6213731.png",
        "https://cdn-icons-png.flaticon.com/512/1055/1055687.png",
        "https://cdn-icons-png.flaticon.com/512/2920/2920244.png",
        "https://cdn-icons-png.flaticon.com/512/3063/3063822.png",
        "https://cdn-icons-png.flaticon.com/512/2933/2933978.png",
        "https://cdn-icons-png.flaticon.com/512/2910/2910760.png"
    ],
    "Description": [
        "Analyze complex datasets and build predictive models.",
        "Develop interactive and responsive websites.",
        "Build AI models using deep learning and NLP/Computer Vision.",
        "Design intuitive UI/UX for apps and websites.",
        "Develop ML pipelines and optimize models.",
        "Develop and maintain backend systems and APIs.",
        "Build frontend interfaces with modern frameworks.",
        "Analyze business data and provide actionable insights.",
        "Test software manually and with automation tools.",
        "Deploy cloud infrastructure and manage services.",
        "Protect systems from cyber threats and vulnerabilities."
    ],
    "Learning_Link": [
        "https://www.udemy.com/course/data-science/","https://www.udemy.com/course/web-development/",
        "https://www.udemy.com/course/ai-engineer/","https://www.udemy.com/course/ui-ux/",
        "https://www.udemy.com/course/machine-learning/","https://www.udemy.com/course/backend-development/",
        "https://www.udemy.com/course/frontend-development/","https://www.udemy.com/course/business-analysis/",
        "https://www.udemy.com/course/software-testing/","https://www.udemy.com/course/cloud-computing/",
        "https://www.udemy.com/course/cyber-security/"
    ]
}

df = pd.DataFrame(data)

# ===================== FUNCTIONS =====================
def calculate_similarity(user_input, skills_list):
    vectorizer = TfidfVectorizer()
    skill_matrix = vectorizer.fit_transform(skills_list)
    user_vector = vectorizer.transform([user_input])
    return cosine_similarity(user_vector, skill_matrix).flatten()

def get_missing_skills(user_skills, required_skills_str):
    req_set = set(s.strip().lower() for s in required_skills_str.split(","))
    return req_set - user_skills

def assign_badge(score):
    if score >= 85: return "🏆 Excellent Fit"
    elif score >= 70: return "🔥 Good Fit"
    else: return "⚠️ Needs Improvement"

def plot_radar_chart(top_career, user_skills):
    required_skills = set(skill.strip() for skill in top_career["Required_Skills"].split(","))
    all_skills = list(required_skills.union(user_skills))
    
    # Create vectors for radar
    user_vec = [1 if s.lower() in user_skills else 0 for s in all_skills]
    req_vec = [1 if s in required_skills else 0 for s in all_skills]
    
    radar_df = pd.DataFrame({
        'Skill': all_skills * 2,
        'Value': user_vec + req_vec,
        'Source': ['You']*len(all_skills) + ['Required']*len(all_skills)
    })
    
    fig = px.line_polar(radar_df, r='Value', theta='Skill', color='Source', 
                        line_close=True, title=f"Skill Gap Analysis: {top_career['Career']}")
    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True)

def generate_pdf(recs_df, user_skills):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Career Recommendation Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 10, f"Your Skills: {', '.join(user_skills)}", ln=True)
    pdf.ln(10)
    
    for _, row in recs_df.head(5).iterrows():
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"{row['Career']} ({row['Match_Score']:.1f}%)", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 7, f"Description: {row['Description']}")
        missing = get_missing_skills(user_skills, row['Required_Skills'])
        pdf.multi_cell(0, 7, f"Skills to Learn: {', '.join(missing) if missing else 'None! You are ready.'}")
        pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin-1')

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Career Guide AI", layout="wide", page_icon="🎯")
st.title("🎯 Ultimate Career Recommendation System")

# Sidebar for Searching
with st.sidebar:
    st.header("🔍 Quick Skill Search")
    search = st.text_input("Find careers by specific skill:")
    if search:
        matches = df[df["Required_Skills"].str.lower().str.contains(search.lower())]
        for _, r in matches.iterrows():
            st.info(f"**{r['Career']}**")

# User Interaction
user_input = st.text_area("🧠 Type your skills (e.g., Python, SQL, Figma):", "Python, SQL, HTML")
user_skills_processed = set(s.strip().lower() for s in user_input.split(","))

if st.button("🚀 Analyze My Career Path"):
    # 1. Similarity Calculation
    df["Match_Score"] = calculate_similarity(user_input, df["Required_Skills"]) * 100
    recommendations = df.sort_values(by="Match_Score", ascending=False).reset_index(drop=True)
    
    # 2. Results Header
    st.success(f"Top Recommendation: **{recommendations.iloc[0]['Career']}**")
    
    # 3. Top 3 Details
    st.subheader("🏆 Top 3 Matches")
    cols = st.columns(3)
    for i in range(3):
        row = recommendations.iloc[i]
        with cols[i]:
            st.image(row["Image"], width=80)
            st.markdown(f"### {row['Career']}")
            st.write(assign_badge(row['Match_Score']))
            st.metric("Match Score", f"{row['Match_Score']:.1f}%")
            
            missing = get_missing_skills(user_skills_processed, row['Required_Skills'])
            if missing:
                st.warning(f"Learn: {', '.join(list(missing)[:3])}...")
            else:
                st.success("Perfect Skill Alignment!")

    st.divider()

    # 4. Analytics Section
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📊 Career Match Overview")
        fig_bar = px.bar(recommendations, x="Match_Score", y="Career", 
                         orientation='h', color="Match_Score", template="plotly_dark")
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_right:
        st.subheader("🎯 Deep Dive: Top Career")
        plot_radar_chart(recommendations.iloc[0], user_skills_processed)

    # 5. Career Comparison Tool
    st.divider()
    st.subheader("⚔️ Compare Careers")
    c_list = recommendations["Career"].tolist()
    choice1 = st.selectbox("Career 1", c_list, index=0)
    choice2 = st.selectbox("Career 2", c_list, index=1)
    
    comp_col1, comp_col2 = st.columns(2)
    r1 = df[df["Career"] == choice1].iloc[0]
    r2 = df[df["Career"] == choice2].iloc[0]
    
    with comp_col1:
        st.info(f"**{choice1}**\n\nSkills: {r1['Required_Skills']}")
    with comp_col2:
        st.info(f"**{choice2}**\n\nSkills: {r2['Required_Skills']}")

    # 6. PDF Export
    st.divider()
    st.subheader("📄 Get Your Report")
    pdf_bytes = generate_pdf(recommendations, user_skills_processed)
    st.download_button(label="📥 Download Career Roadmap (PDF)", 
                       data=pdf_bytes, 
                       file_name="My_Career_Roadmap.pdf", 
                       mime="application/pdf")

# Footer
st.markdown("---")
st.caption("Built with ❤️ | Career Recommendation Engine v2.0")