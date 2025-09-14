# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_style("whitegrid")

# -----------------------
# Helper: safe label encoding
# -----------------------
def safe_label_encode(le, series):
    vals = series.astype(str).fillna("NA").to_numpy()
    unseen = [v for v in np.unique(vals) if v not in le.classes_]
    if len(unseen) > 0:
        le.classes_ = np.concatenate([le.classes_, np.array(unseen, dtype=le.classes_.dtype)])
    return le.transform(vals)

def safe_get_options(df, col):
    return sorted(df[col].dropna().unique().tolist()) if (col in df.columns and df[col].notnull().any()) else []

# -----------------------
# Streamlit setup
# -----------------------
st.set_page_config(page_title="Lifestyle Mental Health Recommender", layout="wide")
st.title("🧠 Lifestyle-Based Mental Health Recommender System")
st.write("Upload your survey CSV, explore compact cluster insights, and get personalized recommendations (TXT only).")

uploaded_file = st.file_uploader("📂 Upload your Lifestyle & Wellness Survey CSV", type=["csv"])

if uploaded_file is not None:
    # Load & clean
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    data = df.drop(columns=["Timestamp", "Score", "Country"], errors="ignore").copy()

    # -----------------------
    # Define numeric vs categorical
    # -----------------------
    numeric_cols = [
        "How stressed do you feel due to work?",
        "On a scale of 1–5, how much control do you feel over your time?",
        "How consistent is your sleep schedule?",
        "How often do you take breaks from screen?"
    ]
    numeric_cols = [c for c in numeric_cols if c in data.columns]
    categorical_cols = [c for c in data.columns if c not in numeric_cols]

    # -----------------------
    # Encode categorical training data and keep encoders
    # -----------------------
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

    # -----------------------
    # Scale features
    # -----------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(data_encoded)

    # -----------------------
    # FORCE k = 3 (user requested)
    # -----------------------
    chosen_k = 3
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10).fit(X)
    df["Cluster"] = kmeans.labels_

    # -----------------------
    # Recommendations & descriptions (customize as needed)
    # -----------------------
    recommendations = {
        0: [
            "Try daily 10–15 min guided meditation and breathing exercises.",
            "Reduce screen exposure 30–60 minutes before bed to improve sleep.",
            "Aim for 30 minutes of physical activity most days and keep hydrated."
        ],
        1: [
            "Keep maintaining routines, exercise, and social connections.",
            "Add short mindfulness or journaling sessions to build resilience.",
            "Monitor screen time and take frequent short breaks."
        ],
        2: [
            "Use Pomodoro or scheduled breaks to reduce continuous screen time.",
            "Introduce outdoor activity 3× per week and limit multitasking.",
            "Practice short relaxation exercises during stressful periods."
        ]
    }

    cluster_descriptions = {
        0: "🟢 Cluster 0: Higher stress & poorer sleep habits.",
        1: "🔵 Cluster 1: Balanced lifestyle with healthier routines.",
        2: "🟠 Cluster 2: Moderate issues related to screen-heavy behavior."
    }

    # -----------------------
    # Sidebar navigation
    # -----------------------
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Insights", "🧾 Cluster Summaries"])

    # -----------------------
    # HOME PAGE
    # -----------------------
    if page == "🏠 Home":
        st.header("📋 Lifestyle Questionnaire (compact)")

        # Sliders for numeric fields
        slider_values = {}
        for col in numeric_cols:
            default = int(data[col].median()) if col in data.columns else 3
            slider_values[col] = st.slider(col + " (1=Low, 5=High)", 1, 5, default, key=f"sl_{col}")

        # Categorical inputs (radio) and text for activity/self-care
        user_inputs = {}
        for col in categorical_cols:
            if "activity" in col.lower() or "self" in col.lower():
                user_inputs[col] = st.text_input(col, key=f"txt_{col}")
            else:
                opts = safe_get_options(df, col)
                if opts:
                    user_inputs[col] = st.radio(col, opts, key=f"rad_{col}")
                else:
                    user_inputs[col] = st.text_input(col, key=f"txtfallback_{col}")

        # Build user dict aligned with training data columns
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
                user_dict[c] = data[c].mode().iat[0] if not data[c].isna().all() else 0

        if st.button("Get Recommendations"):
            user_df = pd.DataFrame([user_dict])

            # Safe encode categoricals (extend encoders if unseen)
            user_encoded = user_df.copy()
            for col in categorical_cols:
                if col in user_encoded.columns:
                    le = encoder_dict.get(col)
                    if le is not None:
                        user_encoded[col] = safe_label_encode(le, user_encoded[col])
                    else:
                        tmp_le = LabelEncoder()
                        user_encoded[col] = tmp_le.fit_transform(user_encoded[col].astype(str))

            # Ensure numeric
            for col in numeric_cols:
                if col in user_encoded.columns:
                    user_encoded[col] = pd.to_numeric(user_encoded[col], errors="coerce").fillna(data[col].median())

            # Scale & predict
            user_scaled = scaler.transform(user_encoded)
            user_cluster = kmeans.predict(user_scaled)[0]

            # Show results and TXT download
            st.subheader(f"📌 You belong to Lifestyle Cluster: {user_cluster}")
            st.markdown(cluster_descriptions.get(user_cluster, ""))
            st.success("✅ Personalized Recommendations:")
            rec_text = f"Lifestyle Cluster: {user_cluster}\n{cluster_descriptions.get(user_cluster, '')}\n\nRecommendations:\n"
            for r in recommendations.get(user_cluster, []):
                st.write("- " + r)
                rec_text += "- " + r + "\n"

            st.download_button("⬇️ Download Recommendations (TXT)",
                               data=rec_text,
                               file_name="recommendations.txt",
                               mime="text/plain")

    # -----------------------
    # INSIGHTS PAGE (compact)
    # -----------------------
    elif page == "📊 Insights":
        st.header("📊 Lifestyle Cluster Insights (compact)")

        if "Cluster" in df.columns:
            # two small charts side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Cluster Distribution")
                fig1, ax1 = plt.subplots(figsize=(4, 3))
                df["Cluster"].value_counts().sort_index().plot(kind="bar", ax=ax1,
                                                               color=sns.color_palette("Set2", n_colors=chosen_k))
                ax1.set_xlabel("Cluster"); ax1.set_ylabel("Count")
                st.pyplot(fig1)
            with col2:
                st.subheader("Cluster % Share")
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                df["Cluster"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax2,
                                                  colors=sns.color_palette("Set2", n_colors=chosen_k))
                ax2.set_ylabel("")
                st.pyplot(fig2)

            # silhouette
            try:
                sil = silhouette_score(X, kmeans.labels_)
                st.info(f"Silhouette Score: **{sil:.3f}**")
            except Exception:
                st.warning("Silhouette score not available.")

            # table and heatmap side-by-side
            col3, col4 = st.columns([1, 1])
            cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
            with col3:
                st.subheader("Average (numeric) per Cluster")
                st.dataframe(cluster_profiles.style.format("{:.2f}"), height=200)
            with col4:
                st.subheader("Feature Heatmap")
                fig3, ax3 = plt.subplots(figsize=(4.5, 3))
                sns.heatmap(cluster_profiles.T, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
                st.pyplot(fig3)

            # small boxplot selector
            st.subheader("Feature distribution by cluster")
            numeric_sel = st.selectbox("Choose numeric feature", options=list(cluster_profiles.columns), index=0)
            fig4, ax4 = plt.subplots(figsize=(6, 3))
            sns.boxplot(x="Cluster", y=numeric_sel, data=df, ax=ax4, palette="Set2")
            ax4.set_title(f"{numeric_sel} by Cluster")
            st.pyplot(fig4)
        else:
            st.error("Clustering not available. Check uploaded dataset.")

    # -----------------------
    # CLUSTER SUMMARIES (compact cards)
    # -----------------------
    elif page == "🧾 Cluster Summaries":
        st.header("🧾 Cluster Summaries (compact cards)")
        if "Cluster" in df.columns:
            counts = df["Cluster"].value_counts().sort_index()
            unique_clusters = sorted([0,1,2])  # force display of clusters 0,1,2
            cols = st.columns(3)
            for i, cid in enumerate(unique_clusters):
                with cols[i]:
                    st.markdown(f"### Cluster {cid}")
                    st.write(f"Count: {int(counts.get(cid,0))}")
                    st.markdown(cluster_descriptions.get(cid, "No description available."))
                    st.write("Top recommendations:")
                    for rec in recommendations.get(cid, []):
                        st.write("- " + rec)
        else:
            st.error("Clustering not available. Check uploaded dataset.")
else:
    st.info("⚠️ Please upload your dataset CSV to begin.")
