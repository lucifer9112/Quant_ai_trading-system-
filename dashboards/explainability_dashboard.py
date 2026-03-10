"""Streamlit dashboard for SHAP explainability."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from explainability.shap_explainer import ShapExplainer


class ExplainabilityDashboard:
    """Render global and local SHAP explanations in Streamlit."""

    def __init__(self, explainer=None):
        self.explainer = explainer or ShapExplainer()

    def run(self, model, feature_frame: pd.DataFrame):
        report = self.explainer.explain(model, feature_frame)

        st.title("Model Explainability")
        st.subheader("Global Importance")
        st.bar_chart(report.feature_importance.set_index("feature")["mean_abs_shap"].head(20))

        st.subheader("Local Explanation")
        selected_index = st.selectbox("Observation", list(report.shap_values.index))
        local = report.shap_values.loc[selected_index].abs().sort_values(ascending=False).head(15)
        st.bar_chart(local)

        st.subheader("Feature Snapshot")
        st.dataframe(feature_frame.loc[[selected_index]].T.rename(columns={selected_index: "value"}))
