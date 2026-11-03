# Telco Churn — Streamlit app. It only READS model.joblib, data/ and artifacts/: it never retrains.
import joblib
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
import streamlit as st

st.set_page_config(page_title="Telco Churn", layout="wide")

DATA = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
INTERNET_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                 "TechSupport", "StreamingTV", "StreamingMovies"]


# --- Section 1: loading (cached) -------------------------------------------------------
@st.cache_resource                      # ONE model object shared by every user and every rerun
def load_model():
    return joblib.load("model.joblib")


@st.cache_data                          # a table; each call hands back its own copy
def load_data():
    df = pd.read_csv(DATA)
    # Same cleaning as the notebook, so the app sees the columns exactly as the pipeline did
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)  # tenure 0 -> 0
    df[INTERNET_COLS] = df[INTERNET_COLS].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df


@st.cache_data
def load_artifact(name):
    return pd.read_csv(f"artifacts/{name}.csv", index_col=0)


@st.cache_resource
def load_explainers(_model, data):
    # Both explain the LogReg in the space it really sees: scaled numbers + one-hot columns.
    prep, clf = _model.named_steps["prep"], _model.named_steps["clf"]
    X_t = np.asarray(prep.transform(data[list(_model.feature_names_in_)]))
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]   # "cat__Contract_Two year" -> "Contract_Two year"
    # ponytail: background = all 7043 customers (the notebook used train only); tiny shift in the mean
    lime = LimeTabularExplainer(X_t, feature_names=names, class_names=["Stays", "Leaves"],
                                mode="classification", random_state=42)
    return prep, clf, X_t.mean(axis=0), names, lime


model = load_model()
df = load_data()
prep, clf, x_mean, feat_names, lime_explainer = load_explainers(model, df)

st.title("Telco Churn — retention assistant")
tab_eda, tab_perf, tab_pred, tab_biz = st.tabs(
    ["EDA", "Model performance", "Prediction & explainability", "Business"])


# --- Section 2: EDA -------------------------------------------------------------------
with tab_eda:
    # Controls: every chart and KPI below only uses the customers that pass these filters
    f1, f2, f3 = st.columns(3)
    contracts = f1.multiselect("Contract", sorted(df["Contract"].unique()),
                               default=sorted(df["Contract"].unique()))
    internet = f2.multiselect("Internet service", sorted(df["InternetService"].unique()),
                              default=sorted(df["InternetService"].unique()))
    t_min, t_max = f3.slider("Tenure (months)", 0, int(df["tenure"].max()), (0, int(df["tenure"].max())))

    seg = df[df["Contract"].isin(contracts)
             & df["InternetService"].isin(internet)
             & df["tenure"].between(t_min, t_max)]

    if seg.empty:
        st.warning("No customers match these filters.")
    else:
        churned = seg["Churn"] == "Yes"
        k1, k2, k3 = st.columns(3)
        k1.metric("Customers in segment", f"{len(seg):,}")
        k2.metric("Churn rate", f"{churned.mean():.1%}")
        k3.metric("Monthly revenue lost to churn", f"${seg.loc[churned, 'MonthlyCharges'].sum():,.0f}")

        g1, g2 = st.columns(2)
        g1.plotly_chart(px.histogram(seg, x="tenure", color="Churn", barmode="overlay", nbins=36,
                                     title="Tenure of customers who stay vs leave"),
                        width="stretch")
        g2.plotly_chart(px.histogram(seg, x="Contract", color="Churn", barnorm="percent",
                                     title="Churn share (%) by contract type"),
                        width="stretch")
        st.plotly_chart(px.box(seg, x="Churn", y="MonthlyCharges", color="Churn",
                               title="Monthly charges: stay vs leave"),
                        width="stretch")


# --- Section 3: model performance (everything read from artifacts/, nothing recomputed by training) ---
with tab_perf:
    st.subheader("1. Model comparison (5-fold CV on train)")
    st.dataframe(load_artifact("cv_models").round(3))

    st.subheader("2. Hyperparameter search: same model, same folds, same budget")
    search = load_artifact("search_summary")
    st.dataframe(search)
    s1, s2 = st.columns(2)
    s1.plotly_chart(px.bar(search, y="best_f1", title="Best CV F1 per strategy", range_y=[0.6, 0.65]),
                    width="stretch")
    s2.plotly_chart(px.bar(search, y="time_s", title="Search time (s)"), width="stretch")

    st.subheader("3. Multi-objective search (NSGA-II): recall vs precision")
    front = load_artifact("pareto_front")
    front["chosen"] = front.index == 31          # the point picked by business cost (Findings 13)
    st.plotly_chart(px.scatter(front, x="recall", y="precision", color="chosen",
                               hover_data=["churn_weight", "C", "penalty", "cost"],
                               title="Pareto front (chosen point highlighted)"),
                    width="stretch")

    st.subheader("4. Final model on test (used once, at the end)")
    pred = pd.read_csv("artifacts/test_predictions.csv")
    m = st.columns(5)
    for col, (name, value) in zip(m, [
            ("Accuracy", accuracy_score(pred.y_true, pred.y_pred)),
            ("Precision", precision_score(pred.y_true, pred.y_pred)),
            ("Recall", recall_score(pred.y_true, pred.y_pred)),
            ("F1", f1_score(pred.y_true, pred.y_pred)),
            ("ROC-AUC", roc_auc_score(pred.y_true, pred.proba))]):
        col.metric(name, f"{value:.3f}")
    cm = confusion_matrix(pred.y_true, pred.y_pred)
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred: stays", "Pred: leaves"],
                              y=["Real: stays", "Real: leaves"], color_continuous_scale="Blues",
                              title="Confusion matrix (test)"),
                    width="stretch")


# --- Section 4a: customer form -> prediction ---------------------------------------------
with tab_pred:
    # The pipeline expects the SAME 19 raw columns, in the SAME order and with the SAME types it saw in fit.
    # So the form is built FROM the pipeline: one widget per column it was trained on.
    cols = list(model.feature_names_in_)
    with st.form("customer"):
        st.write("Describe the customer (defaults = the most common customer):")
        grid = st.columns(3)
        row = {}
        for i, col in enumerate(cols):
            box = grid[i % 3]
            if df[col].dtype == object or df[col].nunique() <= 4:      # categories (incl. SeniorCitizen 0/1)
                options = sorted(df[col].unique())
                row[col] = box.selectbox(col, options, index=options.index(df[col].mode()[0]))
            else:                                                     # tenure, MonthlyCharges, TotalCharges
                row[col] = box.number_input(col, min_value=0.0, value=float(df[col].median()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        customer = pd.DataFrame([row], columns=cols)                  # 1 row x 19 columns, pipeline order
        p = model.predict_proba(customer)[0, 1]                       # column 1 = P(churn)
        leaves = model.predict(customer)[0] == 1
        c1, c2 = st.columns(2)
        c1.metric("Probability of leaving", f"{p:.0%}")
        c2.metric("Prediction", "Leaves" if leaves else "Stays")

        # --- Section 4b: why? SHAP (exact for a LogReg) and LIME, same customer ---------------------
        x_t = np.asarray(prep.transform(customer))[0]
        shap_vals = clf.coef_[0] * (x_t - x_mean)        # w_j (x_j - mean_j): + pushes to "leaves"
        top = pd.Series(shap_vals, index=feat_names)
        top = top[top.abs().sort_values(ascending=False).index[:8]][::-1]
        lime_exp = lime_explainer.explain_instance(x_t, clf.predict_proba, num_features=8)
        lime_top = pd.Series(dict(lime_exp.as_list(label=1)))[::-1]

        def bars(s, title):
            return px.bar(x=s.values, y=s.index, orientation="h", title=title,
                          color=np.where(s.values > 0, "pushes to LEAVE", "pushes to STAY"),
                          color_discrete_map={"pushes to LEAVE": "#d62728", "pushes to STAY": "#1f77b4"},
                          labels={"x": "contribution", "y": "", "color": ""})

        e1, e2 = st.columns(2)
        e1.plotly_chart(bars(top, "SHAP: what moved this customer away from the average"), width="stretch")
        e2.plotly_chart(bars(lime_top, "LIME: same customer, local approximation"), width="stretch")
        st.info(f"In plain words: the strongest reason is **{top.index[-1]}** "
                f"({'raises' if top.iloc[-1] > 0 else 'lowers'} the risk). "
                "These are associations the model learned, not causes.")
