# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score

sns.set_style("whitegrid")

# -----------------------
# Streamlit setup
# -----------------------
st.set_page_config(page_title="Lifestyle Mental Health Recommender", layout="wide")
st.title("🧠 Lifestyle-Based Mental Health Recommender System")
st.write("Upload your survey CSV, run clustering when ready, and explore compact insights with personalized recommendations.")

# -----------------------
# Helper functions
# -----------------------
def safe_label_encode(le, series):
    vals = series.astype(str).fillna("NA").to_numpy()
    unseen = [v for v in np.unique(vals) if v not in le.classes_]
    if len(unseen) > 0:
        le.classes_ = np.concatenate([le.classes_, np.array(unseen, dtype=le.classes_.dtype)])
    return le.transform(vals)

def safe_get_options(df, col):
    return sorted(df[col].dropna().unique().tolist()) if (col in df.columns and df[col].notnull().any()) else []

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def preprocess_data(df):
    data = df.drop(columns=["Timestamp", "Score", "Country"], errors="ignore").copy()

    numeric_cols = [
        "How stressed do you feel due to work?",
        "On a scale of 1–5, how much control do you feel over your time?",
        "How consistent is your sleep schedule?",
        "How often do you take breaks from screen?"
    ]
    numeric_cols = [c for c in numeric_cols if c in data.columns]
    categorical_cols = [c for c in data.columns if c not in numeric_cols]

    # Convert numeric-like columns
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Impute numeric
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        data[numeric_cols] = pd.DataFrame(
            num_imputer.fit_transform(data[numeric_cols]),
            columns=numeric_cols,
            index=data.index
        )

    # Impute categorical
    if categorical_cols:
        for c in categorical_cols:
            data[c] = data[c].astype(str)
        cat_imputer = SimpleImputer(strategy="most_frequent")
        data[categorical_cols] = pd.DataFrame(
            cat_imputer.fit_transform(data[categorical_cols]),
            columns=categorical_cols,
            index=data.index
        )

    return data, numeric_cols, categorical_cols

@st.cache_resource
def get_scaler_and_encoded(data, categorical_cols, numeric_cols):
    encoder_dict = {}
    data_encoded = data.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        encoder_dict[col] = le
    scaler = StandardScaler()
    X = scaler.fit_transform(data_encoded)
    return X, encoder_dict, scaler

# -----------------------
# Main App
# -----------------------
uploaded_file = st.file_uploader("📂 Upload your Lifestyle & Wellness Survey CSV", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    data, numeric_cols, categorical_cols = preprocess_data(df)
    X, encoder_dict, scaler = get_scaler_and_encoded(data, categorical_cols, numeric_cols)

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

        # Inputs for categoricals
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

        # Build user dict
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

        # Run clustering on demand
        if st.button("Run clustering and get recommendations"):
            with st.spinner("Clustering in progress..."):
                chosen_k = 3
                kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10).fit(X)
                df["Cluster"] = kmeans.labels_

                # Encode user input
                user_df = pd.DataFrame([user_dict])
                user_encoded = user_df.copy()
                for col in categorical_cols:
                    if col in encoder_dict:
                        user_encoded[col] = safe_label_encode(encoder_dict[col], user_encoded[col])
                for col in numeric_cols:
                    user_encoded[col] = pd.to_numeric(user_encoded[col], errors="coerce").fillna(data[col].median())
                user_scaled = scaler.transform(user_encoded)
                user_cluster = kmeans.predict(user_scaled)[0]

                # Show results
                cluster_descriptions = {
                    0: "🟢 Cluster 0: Higher stress & poorer sleep habits.",
                    1: "🔵 Cluster 1: Balanced lifestyle with healthier routines.",
                    2: "🟠 Cluster 2: Moderate issues related to screen-heavy behavior."
                }
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

                st.subheader(f"📌 You belong to Lifestyle Cluster: {user_cluster}")
                st.markdown(cluster_descriptions[user_cluster])
                st.success("✅ Personalized Recommendations:")
                rec_text = f"Lifestyle Cluster: {user_cluster}\n{cluster_descriptions[user_cluster]}\n\nRecommendations:\n"
                for r in recommendations[user_cluster]:
                    st.write("- " + r)
                    rec_text += "- " + r + "\n"

                st.download_button("⬇️ Download Recommendations (TXT)",
                                   data=rec_text,
                                   file_name="recommendations.txt",
                                   mime="text/plain")

    # -----------------------
    # INSIGHTS PAGE
    # -----------------------
    elif page == "📊 Insights":
        st.header("📊 Lifestyle Cluster Insights")
        if "Cluster" not in df.columns:
            st.warning("⚠️ Please run clustering first from Home page.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Cluster Distribution")
                fig1, ax1 = plt.subplots(figsize=(4, 3))
                df["Cluster"].value_counts().sort_index().plot(kind="bar", ax=ax1,
                                                               color=sns.color_palette("Set2", n_colors=3))
                st.pyplot(fig1)
            with col2:
                st.subheader("Cluster % Share")
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                df["Cluster"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90,
                                                  ax=ax2, colors=sns.color_palette("Set2", n_colors=3))
                ax2.set_ylabel("")
                st.pyplot(fig2)

            # Metrics
            sil = silhouette_score(X, df["Cluster"])
            dbi = davies_bouldin_score(X, df["Cluster"])
            st.info(f"Silhouette Score: **{sil:.3f}**, Davies–Bouldin Index: **{dbi:.3f}**")

            # Profiles
            cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
            col3, col4 = st.columns([1, 1])
            with col3:
                st.subheader("Average per Cluster")
                st.dataframe(cluster_profiles.style.format("{:.2f}"), height=200)
            with col4:
                st.subheader("Feature Heatmap")
                fig3, ax3 = plt.subplots(figsize=(4.5, 3))
                sns.heatmap(cluster_profiles.T, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
                st.pyplot(fig3)

    # -----------------------
    # CLUSTER SUMMARIES
    # -----------------------
    elif page == "🧾 Cluster Summaries":
        st.header("🧾 Cluster Summaries")
        if "Cluster" not in df.columns:
            st.warning("⚠️ Please run clustering first from Home page.")
        else:
            counts = df["Cluster"].value_counts().sort_index()
            cols = st.columns(3)
            descriptions = {
                0: "🟢 Cluster 0: Higher stress & poorer sleep habits.",
                1: "🔵 Cluster 1: Balanced lifestyle with healthier routines.",
                2: "🟠 Cluster 2: Moderate issues related to screen-heavy behavior."
            }
            recs = {
                0: ["Try daily 10–15 min meditation.", "Reduce screen time before bed.", "Exercise regularly."],
                1: ["Maintain routines.", "Add mindfulness.", "Monitor screen time."],
                2: ["Take Pomodoro breaks.", "Outdoor activity weekly.", "Relaxation exercises."]
            }
            for i, cid in enumerate([0, 1, 2]):
                with cols[i]:
                    st.markdown(f"### Cluster {cid}")
                    st.write(f"Count: {int(counts.get(cid,0))}")
                    st.markdown(descriptions[cid])
                    st.write("Top recommendations:")
                    for rec in recs[cid]:
                        st.write("- " + rec)
else:
    st.info("⚠️ Please upload your dataset CSV to begin.")
