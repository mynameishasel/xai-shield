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

    risk = (
        X.failed_connections * 2.2
        + (X.packet_count > 750) * 5
        + (X.connection_duration < 3) * 3
        + (X.bytes_sent > 40000) * 2
    )

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
        1,
        10000,
        400
    )

    duration = st.number_input(
        "Connection duration (seconds)",
        0.1,
        3600.0,
        30.0
    )

    failed = st.number_input(
        "Failed connections",
        0,
        100,
        2
    )

    sent = st.number_input(
        "Bytes sent",
        0,
        1000000,
        12000
    )

    received = st.number_input(
        "Bytes received",
        0,
        1000000,
        18000
    )

    analyze = st.button(
        "Analyze Activity",
        type="primary"
    )


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

    prediction = int(
        model.predict(sample)[0]
    )

    probabilities = model.predict_proba(
        sample
    )[0]

    probability = float(
        probabilities[prediction]
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


    st.divider()


    # =====================================================
    # SHAP EXPLANATION
    # =====================================================
    st.header("SHAP Explanation")

    st.write(
        "SHAP explains how each network feature influenced this "
        "specific prediction."
    )


    shap_values = shap_explainer(
        sample
    )

    values = shap_values.values


    # Handle different SHAP output shapes
    if values.ndim == 3:

        class_values = values[
            0,
            :,
            prediction
        ]

    elif values.ndim == 2:

        class_values = values[0]

    else:

        class_values = values.flatten()


    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP Value": class_values
    })


    shap_df["Absolute Impact"] = (
        shap_df["SHAP Value"].abs()
    )


    shap_df = shap_df.sort_values(
        "Absolute Impact",
        ascending=False
    )


    # =====================================================
    # SHAP CLASSIFICATION MESSAGE
    # =====================================================
    if prediction == 1:

        st.error(
            "⚠️ This network activity was classified as MALICIOUS."
        )

    else:

        st.success(
            "✅ This network activity was classified as NORMAL."
        )


    st.write(
        "Positive SHAP values push the model toward **MALICIOUS**. "
        "Negative SHAP values push the model toward **NORMAL**."
    )


    # =====================================================
    # SHAP CHART
    # =====================================================
    shap_chart = shap_df[
        ["Feature", "SHAP Value"]
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


    # =====================================================
    # SHAP ANALYST SUMMARY
    # =====================================================
    st.subheader(
        "SHAP Analyst Summary"
    )


    top_shap = shap_df.head(3)


    for _, row in top_shap.iterrows():

        feature = DISPLAY_NAMES[
            row["Feature"]
        ]

        value = row["SHAP Value"]


        if value > 0:

            st.write(
                f"🔴 **{feature}** — pushed the prediction "
                f"toward **MALICIOUS** "
                f"(SHAP: {value:.4f})"
            )

        elif value < 0:

            st.write(
                f"🟢 **{feature}** — pushed the prediction "
                f"toward **NORMAL** "
                f"(SHAP: {value:.4f})"
            )

        else:

            st.write(
                f"⚪ **{feature}** — had little or no influence."
            )


    strongest_shap = shap_df.iloc[0]

    strongest_shap_name = DISPLAY_NAMES[
        strongest_shap["Feature"]
    ]


    st.info(
        f"According to SHAP, the most influential feature for this "
        f"prediction was **{strongest_shap_name}**."
    )


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
            use_container_width=True,
            hide_index=True
        )


    st.divider()


    # =====================================================
    # LIME EXPLANATION
    # =====================================================
    st.header("LIME Explanation")

    st.write(
        "LIME creates a local approximation of the model around "
        "this network event and identifies the conditions that "
        "most influenced the prediction."
    )


    lime_result = lime_explainer.explain_instance(
        sample.iloc[0].values,
        model.predict_proba,
        num_features=len(FEATURES),
        labels=(prediction,)
    )


    lime_list = lime_result.as_list(
        label=prediction
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


    #    # =====================================================
    # LIME CHART
    # =====================================================
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

    # =====================================================
    # LIME ANALYST SUMMARY
    # =====================================================
    st.subheader(
        "LIME Analyst Summary"
    )


    top_lime = lime_df.head(3)


    for _, row in top_lime.iterrows():

        condition = row[
            "Condition"
        ]

        weight = row[
            "LIME Weight"
        ]


        if weight > 0:

            st.write(
                f"🔴 **{condition}** — supported the "
                f"**{label}** prediction "
                f"(LIME: {weight:.4f})"
            )

        elif weight < 0:

            st.write(
                f"🟢 **{condition}** — opposed the "
                f"**{label}** prediction "
                f"(LIME: {weight:.4f})"
            )

        else:

            st.write(
                f"⚪ **{condition}** — had little or no influence."
            )


    strongest_lime = lime_df.iloc[0]


    st.info(
        "According to LIME, the strongest local condition was "
        f"**{strongest_lime['Condition']}**."
    )


    with st.expander(
        "View technical LIME details"
    ):

        st.dataframe(
            lime_df,
            use_container_width=True,
            hide_index=True
        )


    st.divider()


    # =====================================================
    # SHAP VS LIME COMPARISON
    # =====================================================
    st.header(
        "SHAP vs. LIME"
    )


    st.write(
        "Both techniques explain the same Random Forest prediction, "
        "but they use different approaches."
    )


    comparison_col1, comparison_col2 = st.columns(
        2
    )


    with comparison_col1:

        st.subheader("SHAP")

        st.write(
            "**Approach:** Uses the trained model to estimate "
            "how much each feature contributed to the prediction."
        )

        st.write(
            f"**Strongest feature:** {strongest_shap_name}"
        )

        st.write(
            "**Interpretation:** Shows whether each feature pushed "
            "the prediction toward malicious or normal."
        )


    with comparison_col2:

        st.subheader("LIME")

        st.write(
            "**Approach:** Creates a simplified local model around "
            "this specific network event."
        )

        st.write(
            "**Strongest condition:** "
            f"{strongest_lime['Condition']}"
        )

        st.write(
            "**Interpretation:** Shows which local feature conditions "
            "supported or opposed the selected prediction."
        )


    st.caption(
        "SHAP and LIME values are calculated differently, so their "
        "numeric values should not be directly compared as if they "
        "were on the same scale."
    )


    # =====================================================
    # ANALYZED VALUES
    # =====================================================
    st.divider()

    st.header(
        "Analyzed Network Values"
    )


    display_sample = sample.rename(
        columns=DISPLAY_NAMES
    )


    st.dataframe(
        display_sample,
        use_container_width=True,
        hide_index=True
    )


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