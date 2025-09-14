import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------
# Helper: safe label encoding
# -----------------------
def safe_label_encode(le, series):
    """
    Safely transform a pandas Series using a fitted LabelEncoder `le`.
    If a value is unseen, append it to le.classes_ so transform succeeds.
    """
    vals = series.astype(str).fillna("NA").to_numpy()
    unseen = [v for v in np.unique(vals) if v not in le.classes_]
    if len(unseen) > 0:
        le.classes_ = np.concatenate([le.classes_, np.array(unseen, dtype=le.classes_.dtype)])
    return le.transform(vals)

# -----------------------
# Streamlit setup
# -----------------------
st.set_page_config(page_title="Lifestyle Mental Health Recommender", layout="wide")
st.title(" Lifestyle-Based Mental Health Recommender System")
st.write("Upload your survey CSV, explore cluster insights, and get personalized lifestyle recommendations.")

# -----------------------
# File uploader
# -----------------------
uploaded_file = st.file_uploader(" Upload your Lifestyle & Wellness Survey CSV", type=["csv"])

def safe_get_options(df, col):
    return sorted(df[col].dropna().unique().tolist()) if (col in df.columns and df[col].notnull().any()) else []

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Drop irrelevant columns
    data = df.drop(columns=["Timestamp", "Score", "Country"], errors="ignore").copy()

    # Explicit numeric vs categorical
    numeric_cols = [
        "How stressed do you feel due to work?",
        "On a scale of 1–5, how much control do you feel over your time?",
        "How consistent is your sleep schedule?",
        "How often do you take breaks from screen?"
    ]
    numeric_cols = [c for c in numeric_cols if c in data.columns]
    categorical_cols = [c for c in data.columns if c not in numeric_cols]

    # Encode categoricals
    encoder_dict = {}
    data_encoded = data.copy()
    for col in categorical_cols:
        data_encoded[col] = data_encoded[col].astype(str).fillna("NA")
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        encoder_dict[col] = le

    # Ensure numeric columns are numeric
    for col in numeric_cols:
        data_encoded[col] = pd.to_numeric(data_encoded[col], errors="coerce").fillna(data_encoded[col].median())

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(data_encoded)

    # Choose best K (auto silhouette)
    best_k, best_score = None, -1
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        labels = km.labels_
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_k = score, k
    chosen_k = best_k
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10).fit(X)
    df["Cluster"] = kmeans.labels_

    # Recommendations & descriptions
    recommendations = {
        0: [
            "High stress levels detected: Try meditation and relaxation techniques.",
            "Improve sleep quality by reducing screen time before bed.",
            "Maintain hydration and try 30 minutes of daily exercise."
        ],
        1: [
            "Balanced lifestyle: Keep maintaining healthy habits!",
            "Add mindfulness or journaling to strengthen mental resilience.",
            "Ensure social connections for better emotional health."
        ],
        2: [
            "Moderate lifestyle issues detected: Reduce multitasking on devices.",
            "Engage in outdoor physical activity at least 3 times a week.",
            "Take frequent breaks from screens to reduce eye strain."
        ]
    }
    cluster_descriptions = {
        0: "🟢 **Cluster 0:** High Stress & Poor Sleep - Often feel overworked, use screens before bed, may lack rest.",
        1: "🔵 **Cluster 1:** Balanced Lifestyle - Manage stress well, sleep consistently, engage in healthy routines.",
        2: "🟠 **Cluster 2:** Moderate Issues / Screen-Heavy - Exercise sometimes, but spend long hours on screens and multitasking."
    }

    # Sidebar navigation
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Insights", "🧾 Cluster Summaries"])

    # -----------------------
    # Home Page
    # -----------------------
    if page == "🏠 Home":
        st.header("📋 Lifestyle Questionnaire")

        # Numeric sliders
        slider_values = {}
        for col in numeric_cols:
            med = int(data[col].median()) if col in data.columns else 3
            slider_values[col] = st.slider(col + " (1=Low, 5=High)", 1, 5, med)

        # Categorical inputs
        user_inputs = {}
        for col in categorical_cols:
            if "activity" in col.lower() or "self" in col.lower():
                user_inputs[col] = st.text_input(col, "")
            else:
                options = safe_get_options(df, col)
                if options:
                    user_inputs[col] = st.radio(col, options)
                else:
                    user_inputs[col] = st.text_input(col, "")

        # Combine into user dict
        user_dict = {}
        for c in data.columns:
            if c in slider_values:
                user_dict[c] = slider_values[c]
            elif c in user_inputs:
                val = user_inputs[c]
                if val == "" and safe_get_options(df, c):
                    val = safe_get_options(df, c)[0]
                user_dict[c] = val
            else:
                user_dict[c] = data[c].mode()[0] if not data[c].isna().all() else 0

        if st.button("Get Recommendations"):
            user_df = pd.DataFrame([user_dict])

            # Encode categoricals safely
            user_encoded = user_df.copy()
            for col in categorical_cols:
                if col in user_encoded.columns:
                    le = encoder_dict.get(col)
                    if le is not None:
                        user_encoded[col] = safe_label_encode(le, user_encoded[col])

            # Ensure numeric
            for col in numeric_cols:
                if col in user_encoded.columns:
                    user_encoded[col] = pd.to_numeric(user_encoded[col], errors="coerce").fillna(data[col].median())

            # Scale + predict
            user_scaled = scaler.transform(user_encoded)
            user_cluster = kmeans.predict(user_scaled)[0]

            # Show results
            st.subheader(f" You belong to Lifestyle Cluster: {user_cluster}")
            st.markdown(cluster_descriptions.get(user_cluster, ""))
            st.success(" Recommended Actions:")
            rec_text = f"Lifestyle Cluster: {user_cluster}\n"
            rec_text += cluster_descriptions.get(user_cluster, "") + "\n\nRecommendations:\n"
            for r in recommendations.get(user_cluster, []):
                st.write("- " + r)
                rec_text += "- " + r + "\n"

            # TXT download only
            st.download_button("⬇ Download Recommendations (TXT)",
                               data=rec_text,
                               file_name="recommendations.txt",
                               mime="text/plain")

    # -----------------------
    # Insights Page
    # -----------------------
    elif page == " Insights":
        st.header(" Cluster Insights & Evaluation")

        # Cluster distribution
        fig1, ax1 = plt.subplots()
        df["Cluster"].value_counts().sort_index().plot(kind="bar", ax=ax1, color=sns.color_palette("Set2", n_colors=chosen_k))
        ax1.set_title("Cluster Distribution")
        st.pyplot(fig1)

        # Silhouette score
        sil = silhouette_score(X, kmeans.labels_)
        st.write(f"Silhouette Score: **{sil:.3f}**")

        # Average profiles
        st.subheader(" Average Lifestyle Scores per Cluster")
        cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
        st.dataframe(cluster_profiles)

        # Heatmap
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.heatmap(cluster_profiles.T, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
        ax2.set_title("Feature Comparison Heatmap")
        st.pyplot(fig2)

    # -----------------------
    # Cluster Summaries Page
    # -----------------------
    elif page == " Cluster Summaries":
        st.header(" Cluster Descriptions")
        counts = df["Cluster"].value_counts().sort_index()
        for cid in sorted(df["Cluster"].unique()):
            st.markdown(f"### Cluster {cid} — Count: {int(counts.get(cid, 0))}")
            st.markdown(cluster_descriptions.get(cid, ""))
            st.write("Recommendations:")
            for rec in recommendations.get(cid, []):
                st.write("- " + rec)
            st.markdown("---")

else:
    st.info("⚠️ Please upload your dataset CSV to begin.")
