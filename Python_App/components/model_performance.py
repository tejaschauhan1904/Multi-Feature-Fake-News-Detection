"""
model_performance.py
Multi-Feature Fake News Detection Using R Language
Model Evaluation and Comparative Analytics Component
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

try:
    from config import (
        EVALUATION_METRICS_PATH,
        MODEL_COMPARISON_PATH,
        CONFUSION_MATRIX_PATH,
        CHARTS_DIR,
    )
except ImportError:
    from Python_App.config import (
        EVALUATION_METRICS_PATH,
        MODEL_COMPARISON_PATH,
        CONFUSION_MATRIX_PATH,
        CHARTS_DIR,
    )


def render_model_performance():
    """Renders the comprehensive machine learning evaluation page."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="color: #0F172A; font-weight: 700; margin-bottom: 0.3rem;">📈 Machine Learning Model Evaluation</h2>
            <p style="color: #64748B; font-size: 0.95rem;">
                Comparative evaluation across 3 distinct machine learning algorithms trained in R 
                and tested on held-out stratified test samples (80:20 split, seed = 42).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not EVALUATION_METRICS_PATH.exists():
        st.warning("⚠️ Evaluation metrics file not found. Please run the R pipeline (main.R) first.")
        return

    metrics_df = pd.read_csv(EVALUATION_METRICS_PATH)

    # Highlight Best Model
    best_row = metrics_df.sort_values(by=["F1_Score", "ROC_AUC"], ascending=False).iloc[0]

    st.markdown(
        f"""
        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 1.2rem; margin-bottom: 1.2rem;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #1E40AF; text-transform: uppercase;">
                🏆 Best Performing Model (Selected via Balanced F1-Score & ROC-AUC)
            </div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #1E3A8A; margin: 0.3rem 0;">
                {best_row['Model']}
            </div>
            <div style="font-size: 0.9rem; color: #334155;">
                <b>Accuracy:</b> {best_row['Accuracy']*100:.2f}% &nbsp;|&nbsp; 
                <b>F1-Score:</b> {best_row['F1_Score']:.4f} &nbsp;|&nbsp; 
                <b>Precision:</b> {best_row['Precision']:.4f} &nbsp;|&nbsp; 
                <b>Recall (Sensitivity):</b> {best_row['Recall']:.4f} &nbsp;|&nbsp; 
                <b>ROC-AUC:</b> {best_row['ROC_AUC']:.4f}
            </div>
        </div>
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1.5rem; font-size: 0.88rem; color: #475569;">
            💡 <b>Student Guide:</b> For a simple, real-world explanation of how these 3 models work without complicated math, refer to <code>docs/MODELS_EXPLAINED_SIMPLE.md</code>.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Full Metrics Table
    st.markdown("### 📊 Comprehensive Performance Metrics Table")
    display_df = metrics_df.copy()
    display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
    display_df["Precision"] = display_df["Precision"].apply(lambda x: f"{x*100:.2f}%")
    display_df["Recall"] = display_df["Recall"].apply(lambda x: f"{x*100:.2f}%")
    display_df["Specificity"] = display_df["Specificity"].apply(lambda x: f"{x*100:.2f}%")
    display_df["F1_Score"] = display_df["F1_Score"].apply(lambda x: f"{x:.4f}")
    display_df["ROC_AUC"] = display_df["ROC_AUC"].apply(lambda x: f"{x:.4f}")

    st.dataframe(
        display_df[["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1_Score", "ROC_AUC", "TP", "TN", "FP", "FN"]],
        hide_index=True,
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Comparison Chart
    st.markdown("### 📊 Interactive Metrics Comparison")
    melted = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"],
        var_name="Metric",
        value_name="Score"
    )

    fig_comp = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=["#1E3A8A", "#0284C7", "#059669", "#F59E0B", "#8B5CF6"]
    )
    fig_comp.update_layout(
        yaxis=dict(range=[0.75, 1.05], title="Score (0 - 1.0)"),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=380,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Confusion Matrix Heatmaps
    st.markdown("### 🔲 Confusion Matrices by Architecture")
    cm_cols = st.columns(3)

    for i, row in metrics_df.iterrows():
        with cm_cols[i % 3]:
            model_name = row["Model"]
            tp, tn, fp, fn = row["TP"], row["TN"], row["FP"], row["FN"]
            
            z_vals = [[tn, fp], [fn, tp]]
            fig_cm = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=["Pred Fake", "Pred Real"],
                y=["Actual Fake", "Actual Real"],
                text=[[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]],
                texttemplate="%{text}",
                textfont={"size": 13, "color": "white"},
                colorscale="Blues",
                showscale=False
            ))
            fig_cm.update_layout(
                title=dict(text=model_name, font=dict(size=13, weight="bold")),
                height=260,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Display High-Resolution Generated ggplot2 Visualizations
    st.markdown("### 🖼️ High-Resolution Academic Plots (Generated via ggplot2)")
    
    chart_tabs = st.tabs(["ROC Curves", "Feature Importance", "Model Bar Chart"])

    with chart_tabs[0]:
        roc_img = CHARTS_DIR / "roc_curves.png"
        if roc_img.exists():
            st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC) Curves with AUC Scores", use_container_width=True)
        else:
            st.info("ROC Curve chart not generated yet. Run 11_generate_visualizations.R.")

    with chart_tabs[1]:
        fi_img = CHARTS_DIR / "feature_importance.png"
        if fi_img.exists():
            st.image(str(fi_img), caption="Top Discriminative Features by Mean Decrease Gini in Random Forest", use_container_width=True)
        else:
            st.info("Feature importance chart not generated yet.")

    with chart_tabs[2]:
        bar_img = CHARTS_DIR / "model_comparison_bar.png"
        if bar_img.exists():
            st.image(str(bar_img), caption="Model Performance Comparison across Held-out Test Set", use_container_width=True)
        else:
            st.info("Model comparison bar chart not generated yet.")
