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
    vals = series.astype(str).fillna("NA").to_numpy()
    unseen = [v for v in np.unique(vals) if v not in le.classes_]
    if len(unseen) > 0:
        le.classes_ = np.concatenate([le.classes_, np.array(unseen, dtype=le.classes_.dtype)])
    return le.transform(vals)

# -----------------------
# Streamlit setup
# -----------------------
st.set_page_config(page_title="Lifestyle Mental Health Recommender", layout="wide")
st.title("🧠 Lifestyle-Based Mental Health Recommender System")
st.write("Upload your survey CSV, explore cluster insights, and get personalized lifestyle recommendations.")

uploaded_file = st.file_uploader("📂 Upload your Lifestyle & Wellness Survey CSV", type=["csv"])

def safe_get_options(df, col):
    return sorted(df[col].dropna().unique().tolist()) if (col in df.columns and df[col].notnull().any()) else []

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    data = df.drop(columns=["Timestamp", "Score", "Country"], errors="ignore").copy()

    # Numeric & categorical
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

    # Ensure numeric
    for col in numeric_cols:
        data_encoded[col] = pd.to_numeric(data_encoded[col], errors="coerce").fillna(data_encoded[col].median())

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(data_encoded)

    # Auto choose k (fallback to 3 if fails)
    chosen_k, best_score = 3, -1
    try:
        for k in range(2, 6):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            labels = km.labels_
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score, chosen_k = score, k
    except Exception:
        chosen_k = 3

    # Final clustering
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10).fit(X)
    df["Cluster"] = kmeans.labels_

    # Recommendations & descriptions
    recommendations = {
        0: ["High stress: Try meditation and relaxation.", "Reduce screen time before bed.", "Exercise & hydrate."],
        1: ["Balanced lifestyle: Maintain habits.", "Add mindfulness/journaling.", "Ensure social connections."],
        2: ["Reduce multitasking & screen use.", "Exercise outdoors 3× per week.", "Take frequent screen breaks."]
    }
    cluster_descriptions = {
        0: "🟢 **Cluster 0:** High Stress & Poor Sleep.",
        1: "🔵 **Cluster 1:** Balanced Lifestyle.",
        2: "🟠 **Cluster 2:** Moderate Issues / Screen-Heavy."
    }

    # Sidebar navigation
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Insights", "🧾 Cluster Summaries"])

    # -----------------------
    # Insights Page
    # -----------------------
    if page == "📊 Insights":
        st.header("📊 Lifestyle Cluster Insights")

        if "Cluster" in df.columns:
            # Distribution
            st.subheader("Cluster Distribution")
            fig, ax = plt.subplots()
            df["Cluster"].value_counts().sort_index().plot(kind="bar", ax=ax, color=sns.color_palette("Set2", n_colors=chosen_k))
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Silhouette score
            try:
                sil = silhouette_score(X, kmeans.labels_)
                st.write(f"Silhouette Score: **{sil:.3f}**")
            except:
                st.warning("Silhouette score not available.")

            # Average profiles
            st.subheader("📈 Average Lifestyle Scores per Cluster")
            cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
            st.dataframe(cluster_profiles)

            # Heatmap
            st.subheader("🔥 Feature Comparison")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            sns.heatmap(cluster_profiles.T, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)
        else:
            st.error("No clusters found. Please check dataset.")

    # -----------------------
    # Cluster Summaries Page
    # -----------------------
    elif page == "🧾 Cluster Summaries":
        st.header("🔍 Cluster Summaries")

        if "Cluster" in df.columns:
            counts = df["Cluster"].value_counts().sort_index()
            for cid in sorted(df["Cluster"].unique()):
                st.markdown(f"### Cluster {cid} — Count: {int(counts.get(cid, 0))}")
                st.markdown(cluster_descriptions.get(cid, f"Cluster {cid} description not available."))
                st.write("Recommendations:")
                for rec in recommendations.get(cid, []):
                    st.write("- " + rec)
                st.markdown("---")
        else:
            st.error("No clusters found. Please check dataset.")

    # -----------------------
    # Home Page (unchanged from before)
    # -----------------------
    if page == "🏠 Home":
        st.header("📋 Lifestyle Questionnaire")

        # Sliders
        slider_values = {}
        for col in numeric_cols:
            med = int(data[col].median()) if col in data.columns else 3
            slider_values[col] = st.slider(col + " (1=Low, 5=High)", 1, 5, med)

        # Categorical
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

            # Encode safely
            user_encoded = user_df.copy()
            for col in categorical_cols:
                if col in user_encoded.columns:
                    le = encoder_dict.get(col)
                    if le is not None:
                        user_encoded[col] = safe_label_encode(le, user_encoded[col])

            for col in numeric_cols:
                if col in user_encoded.columns:
                    user_encoded[col] = pd.to_numeric(user_encoded[col], errors="coerce").fillna(data[col].median())

            user_scaled = scaler.transform(user_encoded)
            user_cluster = kmeans.predict(user_scaled)[0]

            st.subheader(f"📌 You belong to Lifestyle Cluster: {user_cluster}")
            st.markdown(cluster_de_
