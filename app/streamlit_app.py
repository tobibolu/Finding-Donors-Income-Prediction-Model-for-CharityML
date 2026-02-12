"""Streamlit dashboard for the Donor Propensity Model.

Launch with: streamlit run app/streamlit_app.py
"""

import sys
import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_census, preprocess, split_data
from src.models import get_models, compare_models, optimize_model, get_classification_report
from src.fairness import fairness_summary
from src.visualizations import (
    plot_income_distribution,
    plot_feature_distributions,
    plot_model_comparison,
    plot_roc_pr_curves,
    plot_feature_importance,
    plot_fairness_results,
)

st.set_page_config(page_title="Donor Propensity Model", layout="wide")

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    data = load_census()
    X, y = preprocess(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return data, X, y, X_train, X_test, y_train, y_test


data, X, y, X_train, X_test, y_train, y_test = load_data()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Model Comparison", "Best Model", "Fairness Audit"],
)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("Donor Propensity Model")
    st.markdown(
        "Predicting high-income individuals from US Census data to optimize "
        "nonprofit donor outreach."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Records", f"{len(data):,}")
    col2.metric("Features", data.shape[1] - 1)
    col3.metric("High-Income Rate", f"{(data['income'] == '>50K').mean():.1%}")

    st.subheader("Income Distribution")
    plot_income_distribution(data)
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader("Feature Distributions")
    features = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    plot_feature_distributions(data, features)
    st.pyplot(plt.gcf())
    plt.close()

# ---------------------------------------------------------------------------
# Page: Model Comparison
# ---------------------------------------------------------------------------

elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("Training 6 classifiers and comparing test performance.")

    @st.cache_data
    def run_comparison(_X_train, _y_train, _X_test, _y_test):
        return compare_models(_X_train, _y_train, _X_test, _y_test)

    with st.spinner("Training models (this may take a minute)..."):
        results_df = run_comparison(X_train, y_train, X_test, y_test)

    full = results_df[results_df["sample_frac"] == 1.0][
        ["model", "test_accuracy", "test_fbeta", "train_time"]
    ].sort_values("test_fbeta", ascending=False)

    st.dataframe(full.reset_index(drop=True), use_container_width=True)

    plot_model_comparison(results_df)
    st.pyplot(plt.gcf())
    plt.close()

# ---------------------------------------------------------------------------
# Page: Best Model
# ---------------------------------------------------------------------------

elif page == "Best Model":
    st.title("Optimized Gradient Boosting")
    st.markdown("Hyperparameter tuning with GridSearchCV.")

    @st.cache_resource
    def run_optimization(_X_train, _y_train):
        return optimize_model(_X_train, _y_train)

    with st.spinner("Optimizing hyperparameters..."):
        best_model, grid = run_optimization(X_train, y_train)

    st.subheader("Best Parameters")
    st.json(grid.best_params_)

    report = get_classification_report(best_model, X_test, y_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{report['accuracy']:.4f}")
    col2.metric("F-beta (0.5)", f"{report['fbeta_0.5']:.4f}")
    col3.metric("ROC AUC", f"{report.get('roc_auc', 'N/A')}")

    st.subheader("ROC & Precision-Recall Curves")
    plot_roc_pr_curves(best_model, X_test, y_test)
    st.pyplot(plt.gcf())
    plt.close()

    st.subheader("Feature Importance (Top 15)")
    plot_feature_importance(best_model, list(X_train.columns), top_n=15)
    st.pyplot(plt.gcf())
    plt.close()

# ---------------------------------------------------------------------------
# Page: Fairness Audit
# ---------------------------------------------------------------------------

elif page == "Fairness Audit":
    st.title("Fairness & Bias Analysis")
    st.markdown(
        "Evaluating the model for algorithmic fairness across race and sex."
    )

    @st.cache_resource
    def get_best_model(_X_train, _y_train):
        best_model, _ = optimize_model(_X_train, _y_train)
        return best_model

    with st.spinner("Loading model..."):
        best_model = get_best_model(X_train, y_train)

    y_pred = best_model.predict(X_test)
    fairness = fairness_summary(y_test.values, y_pred, data.iloc[y_test.index])

    st.subheader("Demographic Parity")
    tab1, tab2 = st.tabs(["Sex", "Race"])
    with tab1:
        st.dataframe(fairness.get("sex_demographic_parity"), use_container_width=True)
    with tab2:
        st.dataframe(fairness.get("race_demographic_parity"), use_container_width=True)

    st.subheader("Equalized Odds")
    tab3, tab4 = st.tabs(["Sex", "Race"])
    with tab3:
        st.dataframe(fairness.get("sex_equalized_odds"), use_container_width=True)
    with tab4:
        st.dataframe(fairness.get("race_equalized_odds"), use_container_width=True)

    st.subheader("Fairness Visualizations")
    plot_fairness_results(fairness)
    st.pyplot(plt.gcf())
    plt.close()
