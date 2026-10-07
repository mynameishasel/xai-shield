import streamlit as st
import pandas as pd
import numpy as np
import shap

from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="XAI Shield",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "packet_count",
    "connection_duration",
    "failed_connections",
    "bytes_sent",
    "bytes_received"
]

DISPLAY_NAMES = {
    "packet_count": "Packet Count",
    "connection_duration": "Connection Duration",
    "failed_connections": "Failed Connections",
    "bytes_sent": "Bytes Sent",
    "bytes_received": "Bytes Received"
}


# =========================================================
# TRAIN DEMONSTRATION MODEL
# =========================================================

@st.cache_resource
def train_demo_model():

    rng = np.random.default_rng(42)
    n = 1500

    X = pd.DataFrame({
        "packet_count": rng.integers(1, 1000, n),
        "connection_duration": rng.uniform(0.1, 300, n),
        "failed_connections": rng.integers(0, 15, n),
        "bytes_sent": rng.integers(50, 50000, n),
        "bytes_received": rng.integers(50, 80000, n),
    })

    # Demonstration risk rules
    risk = (
        X["failed_connections"] * 2.2
        + (X["packet_count"] > 750) * 5
        + (X["connection_duration"] < 3) * 3
        + (X["bytes_sent"] > 40000) * 2
    )

    # 0 = NORMAL
    # 1 = MALICIOUS
    y = (risk > 7).astype(int)

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(X, y)

    return model, X


model, background = train_demo_model()


# =========================================================
# SHAP EXPLAINER
# =========================================================

@st.cache_resource
def create_shap_explainer(_model):

    return shap.TreeExplainer(_model)


shap_explainer = create_shap_explainer(model)


# =========================================================
# LIME EXPLAINER
# =========================================================

@st.cache_resource
def create_lime_explainer(_background):

    return LimeTabularExplainer(
        training_data=_background.values,
        feature_names=FEATURES,
        class_names=["NORMAL", "MALICIOUS"],
        mode="classification",
        random_state=42
    )


lime_explainer = create_lime_explainer(background)


# =========================================================
# PAGE HEADER
# =========================================================

st.title("🛡️ XAI Shield")

st.caption(
    "Detect • Explain • Understand"
)

st.write(
    "Student MVP: classify sample network activity as normal or "
    "malicious and use Explainable AI (SHAP and LIME) to explain "
    "why the model made its decision."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Network Activity")

    packet_count = st.number_input(
        "Packet count",
        min_value=1,
        max_value=10000,
        value=400
    )

    duration = st.number_input(
        "Connection duration (seconds)",
        min_value=0.1,
        max_value=3600.0,
        value=30.0
    )

    failed = st.number_input(
        "Failed connections",
        min_value=0,
        max_value=100,
        value=2
    )

    sent = st.number_input(
        "Bytes sent",
        min_value=0,
        max_value=1000000,
        value=12000
    )

    received = st.number_input(
        "Bytes received",
        min_value=0,
        max_value=1000000,
        value=18000
    )

    analyze = st.button(
        "Analyze Activity",
        type="primary"
    )


# =========================================================
# CREATE SAMPLE
# =========================================================

sample = pd.DataFrame(
    [[
        packet_count,
        duration,
        failed,
        sent,
        received
    ]],
    columns=FEATURES
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = int(
        model.predict(sample)[0]
    )

    probabilities = model.predict_proba(
        sample
    )[0]

    probability = float(
        probabilities[prediction]
    )

    malicious_probability = float(
        probabilities[1]
    )

    label = (
        "MALICIOUS"
        if prediction == 1
        else "NORMAL"
    )


    # =====================================================
    # CLASSIFICATION RESULTS
    # =====================================================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Classification",
        label
    )

    c2.metric(
        "Model confidence",
        f"{probability:.1%}"
    )

    c3.metric(
        "Model",
        "Random Forest"
    )


    if prediction == 1:

        st.error(
            "⚠️ Potential malicious network activity detected."
        )

    else:

        st.success(
            "✅ Network activity appears normal."
        )


    st.caption(
        f"Estimated malicious probability: "
        f"{malicious_probability:.1%}"
    )

    st.divider()


    # =========================================================
    # SHAP EXPLANATION
    # =========================================================

    st.header("SHAP Explanation")

    st.write(
        "SHAP shows how each network feature affected the model's "
        "assessment of malicious activity."
    )


    shap_values = shap_explainer(
        sample
    )

    values = shap_values.values


    # =========================================================
    # ALWAYS EXPLAIN CLASS 1 = MALICIOUS
    # =========================================================

    if values.ndim == 3:

        malicious_shap_values = values[
            0,
            :,
            1
        ]

    elif values.ndim == 2:

        malicious_shap_values = values[0]

    else:

        malicious_shap_values = values.flatten()


    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP Value": malicious_shap_values
    })


    shap_df["Absolute Impact"] = (
        shap_df["SHAP Value"].abs()
    )


    shap_df = shap_df.sort_values(
        "Absolute Impact",
        ascending=False
    )


    # =========================================================
    # SHAP EXPLANATION MESSAGE
    # =========================================================

    if prediction == 1:

        st.error(
            "⚠️ SHAP is showing why the model found malicious "
            "network characteristics."
        )

    else:

        st.success(
            "✅ SHAP is showing why the malicious risk remained low."
        )


    st.write(
        "🔴 Positive SHAP values increased the model's assessment "
        "of **MALICIOUS** activity. "
        "🟢 Negative SHAP values decreased malicious risk and "
        "supported **NORMAL** activity."
    )


    # =========================================================
    # SHAP CHART
    # =========================================================

    shap_chart = shap_df[
        [
            "Feature",
            "SHAP Value"
        ]
    ].copy()


    shap_chart["Feature"] = (
        shap_chart["Feature"]
        .map(DISPLAY_NAMES)
    )


    shap_chart = shap_chart.sort_values(
        "SHAP Value"
    )


    st.bar_chart(
        shap_chart.set_index("Feature"),
        horizontal=True,
        height=300
    )


    # =========================================================
    # SHAP ANALYST SUMMARY
    # =========================================================

    st.subheader(
        "SHAP Analyst Summary"
    )


    top_shap = shap_df.head(3)


    for _, row in top_shap.iterrows():

        feature = DISPLAY_NAMES[
            row["Feature"]
        ]

        value = float(
            row["SHAP Value"]
        )


        if value > 0:

            st.write(
                f"🔴 **{feature}** — increased malicious risk "
                f"(SHAP: {value:.4f})"
            )


        elif value < 0:

            st.write(
                f"🟢 **{feature}** — decreased malicious risk "
                f"and supported NORMAL activity "
                f"(SHAP: {value:.4f})"
            )


        else:

            st.write(
                f"⚪ **{feature}** — had little or no influence."
            )


    # =========================================================
    # STRONGEST SHAP FEATURE
    # =========================================================

    strongest_shap = shap_df.iloc[0]

    strongest_shap_name = DISPLAY_NAMES[
        strongest_shap["Feature"]
    ]

    strongest_shap_value = float(
        strongest_shap["SHAP Value"]
    )


    if strongest_shap_value > 0:

        shap_direction = (
            "increased the model's assessment of malicious activity"
        )

    elif strongest_shap_value < 0:

        shap_direction = (
            "decreased malicious risk and supported normal activity"
        )

    else:

        shap_direction = (
            "had little or no influence on the prediction"
        )


    st.info(
        f"According to SHAP, the most influential feature was "
        f"**{strongest_shap_name}**, which {shap_direction}."
    )


    # =========================================================
    # TECHNICAL SHAP DETAILS
    # =========================================================

    with st.expander(
        "View technical SHAP details"
    ):

        shap_details = shap_df.copy()

        shap_details["Feature"] = (
            shap_details["Feature"]
            .map(DISPLAY_NAMES)
        )

        st.dataframe(
            shap_details,
            width="stretch",
            hide_index=True
        )


    st.divider()


    # =========================================================
    # LIME EXPLANATION
    # =========================================================

    st.header("LIME Explanation")

    st.write(
        "LIME creates a simplified local explanation around this "
        "network event and shows which conditions increased or "
        "decreased the model's assessment of malicious activity."
    )


    # =========================================================
    # ALWAYS EXPLAIN CLASS 1 = MALICIOUS
    # =========================================================

    lime_result = lime_explainer.explain_instance(
        sample.iloc[0].values,
        model.predict_proba,
        num_features=len(FEATURES),
        labels=(1,)
    )


    lime_list = lime_result.as_list(
        label=1
    )


    lime_df = pd.DataFrame(
        lime_list,
        columns=[
            "Condition",
            "LIME Weight"
        ]
    )


    lime_df["Absolute Impact"] = (
        lime_df["LIME Weight"].abs()
    )


    lime_df = lime_df.sort_values(
        "Absolute Impact",
        ascending=False
    )


    # =========================================================
    # LIME MESSAGE
    # =========================================================

    if prediction == 1:

        st.error(
            "⚠️ LIME is showing which local conditions contributed "
            "to malicious risk."
        )

    else:

        st.success(
            "✅ LIME is showing which local conditions helped keep "
            "the malicious risk low."
        )


    st.write(
        "🔴 Positive LIME weights support **MALICIOUS** activity. "
        "🟢 Negative LIME weights oppose malicious activity and "
        "support **NORMAL** activity."
    )


    # =========================================================
    # LIME CHART
    # =========================================================

    lime_chart = lime_df[
        [
            "Condition",
            "LIME Weight"
        ]
    ].copy()


    lime_chart = lime_chart.sort_values(
        "LIME Weight"
    )


    st.bar_chart(
        lime_chart.set_index("Condition"),
        horizontal=True,
        height=300
    )


    # =========================================================
    # LIME ANALYST SUMMARY
    # =========================================================

    st.subheader(
        "LIME Analyst Summary"
    )


    top_lime = lime_df.head(3)


    for _, row in top_lime.iterrows():

        condition = row[
            "Condition"
        ]

        weight = float(
            row["LIME Weight"]
        )


        if weight > 0:

            st.write(
                f"🔴 **{condition}** — increased support for "
                f"the MALICIOUS class "
                f"(LIME: {weight:.4f})"
            )


        elif weight < 0:

            st.write(
                f"🟢 **{condition}** — decreased support for "
                f"the MALICIOUS class and supported NORMAL activity "
                f"(LIME: {weight:.4f})"
            )


        else:

            st.write(
                f"⚪ **{condition}** — had little or no influence."
            )


    # =========================================================
    # STRONGEST LIME CONDITION
    # =========================================================

    strongest_lime = lime_df.iloc[0]

    strongest_lime_condition = (
        strongest_lime["Condition"]
    )

    strongest_lime_weight = float(
        strongest_lime["LIME Weight"]
    )


    if strongest_lime_weight > 0:

        lime_direction = (
            "increased support for malicious activity"
        )

    elif strongest_lime_weight < 0:

        lime_direction = (
            "decreased support for malicious activity "
            "and supported normal activity"
        )

    else:

        lime_direction = (
            "had little or no influence"
        )


    st.info(
        f"According to LIME, the strongest local condition was "
        f"**{strongest_lime_condition}**, which {lime_direction}."
    )


    # =========================================================
    # TECHNICAL LIME DETAILS
    # =========================================================

    with st.expander(
        "View technical LIME details"
    ):

        st.dataframe(
            lime_df,
            width="stretch",
            hide_index=True
        )


    st.divider()


    # =========================================================
    # SHAP VS. LIME
    # =========================================================

    st.header(
        "SHAP vs. LIME"
    )


    st.write(
        "Both techniques explain the model's assessment of the "
        "**MALICIOUS** class, but they use different approaches."
    )


    comparison_col1, comparison_col2 = st.columns(
        2
    )


    with comparison_col1:

        st.subheader("SHAP")

        st.write(
            "**Approach:** Uses the trained Random Forest model "
            "to estimate how each feature influenced malicious risk."
        )

        st.write(
            f"**Strongest feature:** {strongest_shap_name}"
        )

        st.write(
            "**Interpretation:** Positive values increase malicious "
            "risk, while negative values decrease malicious risk."
        )


    with comparison_col2:

        st.subheader("LIME")

        st.write(
            "**Approach:** Creates a simplified local model around "
            "this specific network event."
        )

        st.write(
            f"**Strongest condition:** "
            f"{strongest_lime_condition}"
        )

        st.write(
            "**Interpretation:** Positive weights support malicious "
            "activity, while negative weights oppose malicious activity."
        )


    st.caption(
        "SHAP and LIME use different explanation methods, so their "
        "numeric values should not be directly compared as if they "
        "were on the same scale."
    )


    # =========================================================
    # ANALYZED NETWORK VALUES
    # =========================================================

    st.divider()

    st.header(
        "Analyzed Network Values"
    )


    display_sample = sample.rename(
        columns=DISPLAY_NAMES
    )


    st.dataframe(
        display_sample,
        width="stretch",
        hide_index=True
    )


# =========================================================
# BEFORE ANALYSIS
# =========================================================

else:

    st.info(
        "Enter network values in the sidebar and select "
        "**Analyze Activity**."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Academic prototype. The included generated training data is "
    "for demonstrating the software workflow, not for production "
    "intrusion detection."
)