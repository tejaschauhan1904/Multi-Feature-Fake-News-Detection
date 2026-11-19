"""
news_detection.py
Multi-Feature Fake News Detection Using R Language
Simple, Interactive Article Testing & Plain-English Explanations
"""

import streamlit as st
import time

try:
    from r_integration import predict_news_article, RIntegrationError
except ImportError:
    from Python_App.r_integration import predict_news_article, RIntegrationError

# Pre-defined test samples for simple demonstration during college viva/presentation
DEMO_SAMPLES = {
    "-- Select an example article to test --": {"title": "", "text": ""},
    "📰 Real News: NASA Climate Observation Satellite": {
        "title": "NASA Successfully Deploys Advanced Global Climate Satellite Into Orbit",
        "text": (
            "The National Aeronautics and Space Administration announced on Tuesday the successful launch "
            "of its next-generation environmental observation spacecraft from Vandenberg Space Force Base. "
            "The mission, developed in partnership with international meteorological agencies, will deliver "
            "precise measurements of atmospheric temperatures, ocean currents, and cloud formations to improve "
            "long-term weather modeling and disaster response preparedness across global meteorological centers."
        ),
    },
    "🚨 Fake News: Secret Elite Conspiracy": {
        "title": "SHOCKING BOMBSHELL: Whistleblower LEAKS Classified Evidence Exposing Secret Plot!",
        "text": (
            "YOU WILL NEVER BELIEVE WHAT THEY JUST DISCOVERED! In a scandalous leaked document that the corrupt "
            "mainstream media is desperately trying to censor, high-ranking globalists were caught red-handed plotting "
            "a massive takeover! Anonymous patriotic sources confirmed that secret orders were signed behind closed "
            "doors last night! Share this viral report everywhere before federal agents take down the website!!!"
        ),
    },
    "📰 Real News: Central Bank Economic Monetary Policy": {
        "title": "Federal Reserve Holds Benchmark Interest Rates Steady Amid Cooling Inflation",
        "text": (
            "Federal Reserve officials concluded their two-day policy summit by holding the benchmark federal funds "
            "rate unchanged at 5.25 to 5.50 percent. Chairman Jerome Powell stated during the press conference that "
            "while consumer price indices have shown moderate deceleration over the past two quarters, policymakers "
            "remain vigilant and will continue monitoring labor market indicators and gross domestic product metrics "
            "prior to considering future monetary easing."
        ),
    },
    "🚨 Fake News: Miracle Health Cure Miracle Coverup": {
        "title": "BANNED REMEDY: Doctors are FURIOUS that this ancient miracle cure was revealed!",
        "text": (
            "Big Pharma is in complete PANIC after an obscure researcher leaked the hidden herbal secret that cures "
            "every known disease in 48 hours flat! Corrupt health officials tried to jail him, but we secured the truth! "
            "Don't let them deceive you anymore! Click the secret link now to protect your family from the government scheme!"
        ),
    },
}


def render_news_detection():
    """Renders the simple, interactive article testing page."""
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h2 style="color: #0F172A; font-weight: 700; margin-bottom: 0.3rem;">🔍 Test a News Article</h2>
            <p style="color: #475569; font-size: 0.95rem;">
                Select a sample article or paste your own headline & text below. 
                Our <b>R machine learning backend</b> will analyze 32 different clues and predict whether it is Real or Fake.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sample selector dropdown
    selected_sample = st.selectbox(
        "💡 Quick Demo: Pick a pre-loaded article to test immediately:",
        list(DEMO_SAMPLES.keys()),
        index=0,
        help="Select a demo article to auto-fill the text below for quick presentation."
    )

    default_title = DEMO_SAMPLES[selected_sample]["title"]
    default_text = DEMO_SAMPLES[selected_sample]["text"]

    # Input Form
    with st.form("article_input_form"):
        title_input = st.text_input(
            "Article Headline (Optional):",
            value=default_title,
            placeholder="e.g. NASA Announces New Space Mission",
            help="Headline of the article. Used to check all-caps and exclamation marks in titles."
        )

        text_input = st.text_area(
            "Article Body Text (Required):",
            value=default_text,
            height=180,
            placeholder="Paste the full text of the article here...",
            help="The full news story for sentiment, readability, and word analysis."
        )

        col_submit, col_clear = st.columns([2, 1])
        with col_submit:
            submit_btn = st.form_submit_button("⚡ Analyze News Article", type="primary", use_container_width=True)
        with col_clear:
            clear_btn = st.form_submit_button("🔄 Clear Input", use_container_width=True)

    if clear_btn:
        st.session_state.pop("last_prediction_result", None)
        st.rerun()

    # Process submission
    if submit_btn:
        if not title_input.strip() and not text_input.strip():
            st.warning("⚠️ Please provide at least a title or text to analyze.")
            return

        active_model_key = st.session_state.get("selected_model_key", "logistic_regression")
        active_model_name = st.session_state.get("selected_model_name", "Logistic Regression (LASSO)")

        with st.spinner(f"R Backend is extracting 32 clues and calculating prediction using {active_model_name}..."):
            start_t = time.time()
            try:
                res = predict_news_article(
                    title=title_input,
                    text=text_input,
                    model_name=active_model_key,
                    timeout=30,
                )
                elapsed_s = round(time.time() - start_t, 2)
                res["inference_time_secs"] = elapsed_s
                st.session_state["last_prediction_result"] = res
                st.session_state["last_title"] = title_input
                st.session_state["last_text"] = text_input
            except RIntegrationError as rie:
                st.error(f"❌ R Integration Error: {str(rie)}")
                return
            except Exception as ex:
                st.error(f"❌ Error: {str(ex)}")
                return

    # Display prediction results if present
    if "last_prediction_result" in st.session_state:
        res = st.session_state["last_prediction_result"]
        pred_label = res["prediction"]
        conf_pct = res["confidence_percent"]
        model_name = res["model_used"]
        real_prob = res["real_news_probability"]
        fake_prob = res["fake_news_probability"]
        elapsed_time = res.get("inference_time_secs", 0.0)
        feats = res.get("extracted_features", {})

        is_real = (pred_label == "REAL NEWS")
        card_class = "result-card-real" if is_real else "result-card-fake"
        title_class = "result-title-real" if is_real else "result-title-fake"
        icon = "✅" if is_real else "🚨"

        st.markdown(
            f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #475569;">
                            Prediction Result
                        </div>
                        <h2 class="{title_class}" style="margin-top: 0.2rem;">
                            {icon} {pred_label}
                        </h2>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.85rem; color: #64748B;">Model Confidence</div>
                        <div style="font-size: 2.2rem; font-weight: 700; color: {'#059669' if is_real else '#DC2626'};">
                            {conf_pct}
                        </div>
                    </div>
                </div>
                <div style="margin-top: 0.8rem; padding-top: 0.6rem; border-top: 1px solid rgba(0,0,0,0.08); font-size: 0.85rem; color: #475569; display: flex; justify-content: space-between; flex-wrap: wrap;">
                    <span><b>Evaluated by R Model:</b> {model_name}</span>
                    <span><b>Response Time:</b> {elapsed_time}s</span>
                    <span><b>Odds:</b> {real_prob*100:.1f}% Real vs {fake_prob*100:.1f}% Fake</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Simple "Why was this flagged?" box for College Presentation
        exclamation_cnt = int(feats.get("exclamation_count", 0))
        upper_pct = feats.get("uppercase_ratio", 0) * 100
        net_sent = int(feats.get("sentiment_score", 0))

        if is_real:
            reasons_html = f"""
            <div class="reason-box-real" style="background-color: #ECFDF5; border: 1.5px solid #059669; border-radius: 8px; padding: 1.1rem; margin-bottom: 1.2rem; color: #064E3B;">
                <div style="font-weight: 800; color: #064E3B; font-size: 0.98rem; margin-bottom: 0.4rem;">
                    💡 Why did the model classify this as REAL NEWS?
                </div>
                <ul style="margin: 0 0 0 1.2rem; padding: 0; color: #064E3B;">
                    <li style="color: #064E3B; margin-bottom: 0.3rem;"><b style="color: #064E3B;">Calm & Factual Tone:</b> Sentiment score is balanced ({net_sent}), typical of objective journalistic reporting.</li>
                    <li style="color: #064E3B; margin-bottom: 0.3rem;"><b style="color: #064E3B;">Standard Professional Punctuation:</b> Only {exclamation_cnt} exclamation mark(s) detected. Real news avoids sensational punctuation.</li>
                    <li style="color: #064E3B;"><b style="color: #064E3B;">Normal Capitalization:</b> Only {upper_pct:.1f}% uppercase letters (no aggressive all-caps shouting).</li>
                </ul>
            </div>
            """
        else:
            reasons_html = f"""
            <div class="reason-box-fake" style="background-color: #FEF2F2; border: 1.5px solid #DC2626; border-radius: 8px; padding: 1.1rem; margin-bottom: 1.2rem; color: #7F1D1D;">
                <div style="font-weight: 800; color: #7F1D1D; font-size: 0.98rem; margin-bottom: 0.4rem;">
                    💡 Why did the model classify this as FAKE NEWS?
                </div>
                <ul style="margin: 0 0 0 1.2rem; padding: 0; color: #7F1D1D;">
                    <li style="color: #7F1D1D; margin-bottom: 0.3rem;"><b style="color: #7F1D1D;">Sensational Punctuation:</b> Found {exclamation_cnt} exclamation mark(s), a classic clickbait pattern.</li>
                    <li style="color: #7F1D1D; margin-bottom: 0.3rem;"><b style="color: #7F1D1D;">Heavy Capitalization:</b> {upper_pct:.1f}% uppercase letters detected, common in urgency-creating hoaxes.</li>
                    <li style="color: #7F1D1D;"><b style="color: #7F1D1D;">High Emotional / Polarized Tone:</b> Net sentiment score ({net_sent}) reflects extreme word choices designed to provoke reaction.</li>
                </ul>
            </div>
            """
        st.markdown(reasons_html, unsafe_allow_html=True)

        # Simple Feature Highlights (4 metrics)
        st.markdown("### 📋 Simple Feature Clues")
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)

        with fcol1:
            st.metric("Total Words", f"{feats.get('word_count', 0):,}", help="Total number of words in this article.")
        with fcol2:
            st.metric("Net Sentiment", f"{net_sent}", delta="Positive" if net_sent > 0 else ("Negative" if net_sent < 0 else "Neutral"), help="Positive words minus negative words.")
        with fcol3:
            flesch = feats.get("flesch_reading_ease", 0)
            read_level = "Easy" if flesch >= 60 else ("Standard" if flesch >= 40 else "Advanced")
            st.metric("Reading Level", f"{flesch:.0f} ({read_level})", help="Flesch Reading Ease score from 0 (very hard) to 100 (very easy).")
        with fcol4:
            st.metric("Capital Letters", f"{upper_pct:.1f}%", help="Percentage of letters that are UPPERCASE.")

        # Keywords Row
        st.markdown("### 🔑 Key Vocabulary Words Spotted")
        keywords = res.get("top_keywords", {})
        if keywords:
            pills_html = "".join([f'<span class="feature-pill">{kw}</span>' for kw in keywords.keys()])
            st.markdown(f'<div style="margin-top: 0.2rem;">{pills_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No unusual keywords detected.")

        # Simple Disclaimer
        st.markdown(
            f"""
            <div class="disclaimer-box">
                <b>College PBL Note:</b> This prediction is based on statistical machine learning patterns learned from 
                past news articles. It checks <i>style, wording, and emotional tone</i>; it does not search live Google facts.
            </div>
            """,
            unsafe_allow_html=True
        )
