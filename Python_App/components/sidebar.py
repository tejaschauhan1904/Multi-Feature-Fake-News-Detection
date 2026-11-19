"""
sidebar.py
Multi-Feature Fake News Detection Using R Language
Sidebar Navigation and System Health Component
"""

import streamlit as st

try:
    from config import RSCRIPT_PATH, AVAILABLE_MODELS
    from r_integration import check_r_environment, check_models_presence
except ImportError:
    from Python_App.config import RSCRIPT_PATH, AVAILABLE_MODELS
    from Python_App.r_integration import check_r_environment, check_models_presence


def render_sidebar():
    """Renders the sidebar navigation and backend status information."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0; text-align: center;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #0F172A; letter-spacing: -0.02em;">
                    📰 TruthPulse AI
                </div>
                <div style="font-size: 0.8rem; color: #64748B; font-weight: 500;">
                    R-Powered Multi-Feature Detection
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Navigation")
        pages = [
            "🏠 Dashboard",
            "🔍 Fake News Detection",
            "📊 Feature Analysis",
            "📈 Model Performance",
            "📁 Dataset Analysis",
            "ℹ️ About Project",
        ]

        if "current_page" not in st.session_state:
            st.session_state["current_page"] = pages[0]

        selected_page = st.radio(
            "Go to page",
            pages,
            index=pages.index(st.session_state["current_page"]) if st.session_state["current_page"] in pages else 0,
            label_visibility="collapsed"
        )
        st.session_state["current_page"] = selected_page

        st.markdown("---")
        st.markdown("### Model Selection")
        selected_model_label = st.selectbox(
            "Backend ML Classifier",
            list(AVAILABLE_MODELS.keys()),
            index=0,
            help="Choose which R machine learning model performs feature weighting and classification."
        )
        st.session_state["selected_model_key"] = AVAILABLE_MODELS[selected_model_label]
        st.session_state["selected_model_name"] = selected_model_label

        st.markdown("---")
        st.markdown("### Backend Status")

        # Check R Engine
        r_ok, r_msg = check_r_environment()
        if r_ok:
            st.markdown(
                '<div style="font-size: 0.85rem; margin-bottom: 0.4rem; color: #0F172A;">'
                '<span class="status-indicator status-green"></span>'
                '<b style="color: #0F172A;">R Backend:</b> Connected (v4.6.1)'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="font-size: 0.85rem; margin-bottom: 0.4rem; color: #0F172A;">'
                '<span class="status-indicator status-red"></span>'
                '<b style="color: #0F172A;">R Backend:</b> Disconnected'
                '</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Error: {r_msg[:80]}")

        # Check Models
        models_ok, missing_list = check_models_presence()
        if models_ok:
            st.markdown(
                '<div style="font-size: 0.85rem; color: #0F172A;">'
                '<span class="status-indicator status-green"></span>'
                '<b style="color: #0F172A;">ML Models:</b> Ready (3 Loaded)'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="font-size: 0.85rem; color: #0F172A;">'
                '<span class="status-indicator status-red"></span>'
                '<b style="color: #0F172A;">ML Models:</b> Missing'
                '</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Missing {len(missing_list)} model files. Run main.R to train.")

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #475569; text-align: center; line-height: 1.5;">
                <b style="color: #0F172A;">College Academic Project</b><br>
                Semester 5 R Language PBL<br>
                R Backend + Python Streamlit
            </div>
            """,
            unsafe_allow_html=True
        )

        return selected_page
