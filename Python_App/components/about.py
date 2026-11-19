"""
about.py
Multi-Feature Fake News Detection Using R Language
Simple, Clear Project Overview & College PBL Presentation Guide
"""

import streamlit as st


def render_about():
    """Renders a simple, clean, and beginner-friendly project guide for College PBL."""
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h2 style="color: #0F172A; font-weight: 700; margin-bottom: 0.3rem;">ℹ️ About This College Project</h2>
            <p style="color: #475569; font-size: 0.95rem;">
                Semester 5 Project-Based Learning (PBL) &bull; <b>Multi-Feature Fake News Detection Using R Language</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎯 What Is This Project in 30 Seconds?")
        st.markdown(
            """
            This project is an AI system that reads a news article and predicts whether it is **REAL** or **FAKE**.
            
            Instead of simply looking at word counts, it acts like a detective by analyzing **4 different types of clues**:
            1. **Emotional Words (Sentiment):** Are there extreme angry/shocking words?
            2. **Reading Difficulty (Readability):** Is it written in short clickbait phrases or formal paragraphs?
            3. **Punctuation & Caps (Linguistics):** Are there multiple exclamation marks (`!`) or ALL-CAPS words?
            4. **Topic Words (TF-IDF):** What specific subject words are used?
            """
        )

        st.markdown("### 🛠️ Why Use Both R Language and Python?")
        st.markdown(
            """
            - **R Language (Backend & Brain):** R is world-class for statistics and data science. All data cleaning, 
              feature calculation, TF-IDF matrix generation, and machine learning models are built 100% in R.
            - **Python Streamlit (Frontend & Face):** Python Streamlit makes it fast and easy to build a clean web 
              interface with interactive charts.
            - **How They Talk:** When you click *"Analyze"*, Python sends the text to R via a temporary JSON file, 
              R computes the prediction, and sends the answer back to Python.
            """
        )

        st.markdown("### 🧠 The 3 Machine Learning Models Compared")
        st.markdown(
            """
            We trained and compared three different algorithms in R on 39,100 clean news articles:
            1. **Logistic Regression with LASSO (Our Winner - 99.53% Accuracy):**
               Like a scorecard that awards points to words and clues. The **LASSO** feature automatically throws away 
               useless words so the model doesn't overfit.
            2. **Random Forest (99.33% Accuracy):**
               Like a panel of 100 decision trees that each examine different clues and vote by majority.
            3. **Naive Bayes (89.60% Accuracy):**
               A classical probability calculator that counts word frequencies using Bayes' theorem.
            """
        )

    with col2:
        st.markdown("### 🎓 Quick Viva Talking Points")
        st.markdown(
            """
            <div style="background-color: #FFFFFF; border: 1.5px solid #CBD5E1; border-radius: 8px; padding: 1.1rem; font-size: 0.88rem; line-height: 1.6; color: #1E293B;">
                <b style="color: #0F172A; font-size: 0.95rem;">If the Examiner Asks:</b><br><br>
                <b style="color: #0F172A;">"Why did LASSO win?"</b><br>
                Because LASSO's L1 penalty removes redundant words, keeping only high-impact clues.<br><br>
                <b style="color: #0F172A;">"What data leakage did you prevent?"</b><br>
                We removed 5,798 duplicates and stripped publisher datelines like <code>WASHINGTON (Reuters) -</code>.<br><br>
                <b style="color: #0F172A;">"Does negative sentiment mean fake?"</b><br>
                No! Real news also reports plane crashes and wars. Fake news stands out by its <i>extremity</i> and sensational caps/punctuation.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📚 Comprehensive Models Guide")
        st.markdown(
            """
            <div style="background-color: #EFF6FF; border: 1.5px solid #93C5FD; border-radius: 8px; padding: 1.1rem; font-size: 0.88rem; color: #1E3A8A;">
                <b style="color: #1E3A8A; font-size: 0.95rem;">📄 Viva Preparation Guide:</b><br><br>
                We have created a simple, student-friendly Markdown guide in your project repository:<br>
                <b><code>docs/MODELS_EXPLAINED_SIMPLE.md</code></b><br><br>
                Read it before your presentation to master every algorithm and answer with confidence!
            </div>
            """,
            unsafe_allow_html=True
        )
