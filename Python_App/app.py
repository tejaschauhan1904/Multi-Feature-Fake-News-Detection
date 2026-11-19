"""
app.py
Multi-Feature Fake News Detection Using R Language
Main Python Streamlit Application Entrypoint
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import streamlit as st
from config import ASSETS_DIR
from components.sidebar import render_sidebar
from components.dashboard import render_dashboard
from components.news_detection import render_news_detection
from components.feature_analysis import render_feature_analysis
from components.model_performance import render_model_performance
from components.dataset_analysis import render_dataset_analysis
from components.about import render_about


def load_css():
    """Loads custom CSS stylesheet into the Streamlit session."""
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    # Streamlit Page Configuration
    st.set_page_config(
        page_title="TruthPulse AI | Multi-Feature Fake News Detection",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject styling
    load_css()

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Route to appropriate page component
    if "Dashboard" in selected_page:
        render_dashboard()
    elif "Detection" in selected_page:
        render_news_detection()
    elif "Feature" in selected_page:
        render_feature_analysis()
    elif "Performance" in selected_page:
        render_model_performance()
    elif "Dataset" in selected_page:
        render_dataset_analysis()
    elif "About" in selected_page:
        render_about()
    else:
        render_dashboard()

    # Global Footer
    st.markdown(
        """
        <div style="margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1.5px solid #CBD5E1; text-align: center; font-size: 0.85rem; color: #475569;">
            <b style="color: #0F172A;">Multi-Feature Fake News Detection Using R Language</b> &bull; College Semester 5 PBL Project &bull; Built with R Backend & Python Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
