# 🎯 Career Guide AI

AI-powered career recommendation system that analyzes your skills and matches them with suitable career paths, highlighting **skill gaps**, **salary insights**, and **visual analytics** — all in a modern, interactive UI built using **Streamlit**.

🚀 **Live Demo:**  
👉 https://careerrecommendation-njzj4nktes5krsobx2dbzq.streamlit.app/

---

## 🚀 Features

- ✅ Skill-based career matching (no black-box ML)
- ✅ Accurate **Match Score (%)** using required vs user skills
- ✅ **Top 3 Career Cards** with:
  - Match score
  - Skill gap analysis
  - Salary estimation
  - Direct learning resources

- ✅ **Interactive visualizations**:
  - Career match bar chart
  - Radar chart (You vs Required skills)
  - Top 3 distribution (Donut chart)
  - Career trend line chart

- ✅ Clean UI with hover cards & progress indicators
- ✅ Fully data-driven (CSV-based)

---

## 🧠 How Matching Works

Career matching is calculated using **set-based skill overlap logic**:

```text
Match Score = (Matched Skills / Required Skills) × 100


```text
Match Score = (Matched Skills / Required Skills) × 100

✔ No TF-IDF

✔ No hidden AI

✔ Transparent & explainable results

🖥️ Tech Stack
Technology	Purpose
Python	Core logic
Streamlit	Web UI
Pandas	Data handling
Plotly	Interactive charts
Scikit-Learn	Utility imports (future extensibility)
📊 Visual Analytics Included

📊 Career Match Overview (Bar Chart)

🎯 Skill Gap Radar Chart

🥇🥈🥉 Top 3 Match Distribution

📈 Top 10 Career Trend

These insights help users understand why a career fits them, not just what fits.

📂 Project Structure
Career-Guide-AI/
│
├── app.py
├── career_dataset_100.csv
├── README.md
└── assets/
    └── images/

📁 Dataset Details

The dataset (career_dataset_100.csv) contains:

Career Title

Required Skills

Description

Salary Range

Learning Resource Link

Career Image

▶️ How to Run Locally
1️⃣ Clone the repository
git clone https://github.com/your-username/career-guide-ai.git
cd career-guide-ai

2️⃣ Install dependencies
pip install streamlit pandas plotly scikit-learn

3️⃣ Run the app
streamlit run app.py

🎨 UI Highlights

Hover-animated career cards

Progress bars for match percentage

Skill badges & gap indicators

Clean modern layout (wide mode)

🔮 Future Enhancements

Resume upload & analysis

Career report PDF export

User login & profile saving

Skill recommendation roadmap

Database-backed career expansion

👨‍💻 Author

Rohit
Final Year Project – Career Guidance System

Built with ❤️ using Streamlit

⭐ Support

If you like this project:

⭐ Star the repository

🍴 Fork & improve

🧠 Suggest new career paths
