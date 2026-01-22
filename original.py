import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data={
    "Career": [
        "Data Scientist",
        "Web Developer",
        "AI Engineer",
        "UI/UX Designer"
    ],
    "Required_Skills": [
        "Python, Machine Learning, Statistics",
        "HTML, CSS, JavaScript , React",
        "Python, Deep Learning, TensorFlow,NLP",
        "Adobe XD, Figma,UX Research, Graphic Design,Creativity"
    ]
}

df=pd.DataFrame(data)

st.title("Career Recommendation System")
st.write("Enter your skills to get career recommendations.")
user_input = st.text_area("Your Skills (comma separated)", "Python, Statistics , HTML")

if st.button("Recommend Careers"):
    vectorizer=TfidfVectorizer()
    skill_matrix=vectorizer.fit_transform(df['Required_Skills'])
    user_vector=vectorizer.transform([user_input])
    similarity_scores=cosine_similarity(user_vector, skill_matrix).flatten()

    df['Match_Score']=similarity_scores
    recommendations=df.sort_values(by='Match_Score', ascending=False).head(3)
    st.subheader("Top Career Recommendations:")
    st.table(recommendations[['Career', 'Match_Score']])

    user_skills=set([skill.strip().lower() for skill in user_input.split(',')])
    for _, row in recommendations.iterrows():
        career=row['Career']
        required_skills=set([skill.strip().lower() for skill in row['Required_Skills'].split(',')])
        missing_skills=required_skills - user_skills
        st.write(f"**{career}** -> Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}")