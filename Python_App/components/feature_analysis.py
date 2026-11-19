"""
feature_analysis.py
Multi-Feature Fake News Detection Using R Language
Simple, Clear Multi-Category Feature Breakdown Component
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_feature_analysis():
    """Renders a simple, visual, and intuitive feature breakdown."""
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h2 style="color: #0F172A; font-weight: 700; margin-bottom: 0.3rem;">📊 Feature Clues Breakdown</h2>
            <p style="color: #475569; font-size: 0.95rem;">
                See the <b>4 types of clues</b> our R backend extracted from the article: 
                Emotion (Sentiment), Reading Level, Punctuation/Capitalization, and Word Counts.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "last_prediction_result" not in st.session_state:
        st.info("💡 Tip: You can test an article in 'Fake News Detection' first, or review the default demo article clues below.")
        feats = {
            "char_count": 520, "word_count": 78, "sentence_count": 4, "unique_words_count": 62,
            "lexical_diversity": 0.795, "avg_words_per_sentence": 19.5, "avg_chars_per_word": 5.6,
            "long_words_count": 28, "short_words_count": 22, "long_word_ratio": 0.359, "short_word_ratio": 0.282,
            "paragraph_count": 2, "avg_paragraph_length": 260.0,
            "uppercase_count": 18, "uppercase_ratio": 0.034, "exclamation_count": 0, "question_count": 0,
            "total_punct_count": 9, "punct_ratio": 0.017, "url_count": 0, "numeric_count": 3,
            "avg_sentence_length_chars": 130.0, "title_upper_ratio": 0.12, "title_exclamations": 0, "title_questions": 0,
            "flesch_reading_ease": 42.5, "pos_words_count": 5, "neg_words_count": 1, "sentiment_score": 4,
            "pos_word_ratio": 0.064, "neg_word_ratio": 0.013, "sentiment_balance": 0.667
        }
        keywords = {"government": 1.25, "economic": 1.18, "policy": 1.05, "growth": 0.92, "inflation": 0.88}
        article_title = "Demo Policy Report"
    else:
        res = st.session_state["last_prediction_result"]
        feats = res.get("extracted_features", {})
        keywords = res.get("top_keywords", {})
        article_title = st.session_state.get("last_title", "Analyzed Article")

    st.markdown(f"**Inspecting Article:** *{article_title or 'Current Article'}*")

    # 4 Simple Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Emotion & Sentiment",
        "2. Reading Difficulty",
        "3. Caps & Punctuation",
        "4. Word Counts & Keywords"
    ])

    # TAB 1: Sentiment
    with tab1:
        st.markdown("### 🎭 Emotion & Sentiment Score")
        st.markdown(
            "Fake news often uses extreme emotional or angry words to trigger an immediate reaction. "
            "Real news tends to use calm, factual, and balanced vocabulary."
        )

        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.metric("Positive Words Found", int(feats.get("pos_words_count", 0)), help="Words like 'successful', 'growth', 'peace'")
        with scol2:
            st.metric("Negative Words Found", int(feats.get("neg_words_count", 0)), help="Words like 'crisis', 'disaster', 'scandal'")
        with scol3:
            st.metric("Net Sentiment Score", int(feats.get("sentiment_score", 0)), help="Positive Words minus Negative Words")

        fig_sent = go.Figure(go.Indicator(
            mode="gauge+number",
            value=feats.get("sentiment_score", 0),
            title={"text": "Net Emotion Meter (Positive vs Negative)"},
            gauge={
                "axis": {"range": [-10, 10]},
                "bar": {"color": "#1E3A8A"},
                "steps": [
                    {"range": [-10, -2], "color": "#FEE2E2"},
                    {"range": [-2, 2], "color": "#F1F5F9"},
                    {"range": [2, 10], "color": "#DCFCE7"},
                ],
            }
        ))
        fig_sent.update_layout(height=240, margin=dict(l=15, r=15, t=30, b=15))
        st.plotly_chart(fig_sent, use_container_width=True)

    # TAB 2: Readability
    with tab2:
        st.markdown("### 📖 How Easy is it to Read? (Flesch Score)")
        st.markdown(
            "Real journalism is written in professional paragraphs with longer sentences (college reading level: 30–50 score). "
            "Fake news is often written in short, punchy phrases to spread fast on social media (easier reading level: 60–80 score)."
        )

        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.metric("Flesch Score", f"{feats.get('flesch_reading_ease', 0):.0f} / 100")
        with rcol2:
            st.metric("Avg Words per Sentence", f"{feats.get('avg_words_per_sentence', 0):.1f}")
        with rcol3:
            st.metric("Avg Letters per Word", f"{feats.get('avg_chars_per_word', 0):.1f}")

        flesch_val = feats.get("flesch_reading_ease", 0)
        level_label = "Easy / Conversational (Often Fake)" if flesch_val >= 60 else "Standard Newspaper / Formal (Often Real)"
        st.info(f"💡 **Reading Level Result:** This article is **{level_label}**.")

    # TAB 3: Punctuation & Caps
    with tab3:
        st.markdown("### 🔣 Capital Letters & Punctuation Habits")
        st.markdown(
            "Real news outlets follow strict editorial guidelines (Reuters, AP): they rarely use exclamation marks `!` "
            "and never write headlines in ALL-CAPS. Fake news relies on all-caps and `!` to grab attention."
        )

        punc_data = pd.DataFrame({
            "Clue": ["Exclamation Marks (!)", "Question Marks (?)", "Total Punctuation Marks", "Capital Letters Ratio"],
            "Article Value": [
                f"{int(feats.get('exclamation_count', 0))}",
                f"{int(feats.get('question_count', 0))}",
                f"{int(feats.get('total_punct_count', 0))}",
                f"{feats.get('uppercase_ratio', 0) * 100:.1f}%"
            ],
            "Normal Real News Pattern": [
                "0 to 1 mark",
                "0 to 1 mark",
                "Standard commas/periods only",
                "3% to 6% (first letters only)"
            ]
        })
        st.dataframe(punc_data, hide_index=True, use_container_width=True)

    # TAB 4: Words & Keywords
    with tab4:
        st.markdown("### 📝 Word Statistics & Top Keywords")
        wcol1, wcol2 = st.columns(2)

        with wcol1:
            st.write(f"• **Total Word Count:** {int(feats.get('word_count', 0)):,}")
            st.write(f"• **Total Characters:** {int(feats.get('char_count', 0)):,}")
            st.write(f"• **Unique Words (Vocabulary):** {int(feats.get('unique_words_count', 0)):,}")
            st.write(f"• **Vocabulary Richness:** {feats.get('lexical_diversity', 0):.2f}")

        with wcol2:
            st.markdown("<b>Prominent Keywords Spotted:</b>", unsafe_allow_html=True)
            if keywords:
                for k in list(keywords.keys())[:6]:
                    st.write(f"• `{k}`")
            else:
                st.write("No distinct keywords.")
