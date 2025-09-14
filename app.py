# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------
# Streamlit config
# -----------------------
st.set_page_config(page_title="Lifestyle Mental Health Recommender", layout="wide")
st.title("🧠 Lifestyle-Based Mental Health Recommender System")
st.write("Upload your survey CSV, explore cluster insights, and get personalized lifestyle recommendations.")

# -----------------------
# File uploader
# -----------------------
uploaded_file = st.file_uploader("📂 Upload your Lifestyle & Wellness Survey CSV", type=["csv"])

def safe_get_options(df, col):
    return sorted(df[col].dropna().unique().tolist()) if (col in df.columns and df[col].notnull().any()) else []

if uploaded_file is not None:
    # Load & clean columns
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.sidebar.markdown("**Dataset preview**")
    if st.sidebar.checkbox("Show raw data (first 10 rows)"):
        st.sidebar.dataframe(df.head(10))

    # Drop irrelevant columns if exist
    data = df.drop(columns=["Timestamp", "Score", "Country"], errors="ignore").copy()

    # -----------------------
    # Explicit numeric vs categorical feature lists
    # -----------------------
    # Adjust these names if your CSV columns are slightly different; they must match exactly.
    numeric_cols = [
        "How stressed do you feel due to work?",
        "On a scale of 1–5, how much control do you feel over your time?",
        "How consistent is your sleep schedule?",
        "How often do you take breaks from screen?"
    ]
    # Ensure numeric_cols present in data
    numeric_cols = [c for c in numeric_cols if c in data.columns]

    categorical_cols = [c for c in data.columns if c not in numeric_cols]

    # -----------------------
    # Encode categorical columns (LabelEncoder)
    # -----------------------
    encoder_dict = {}
    data_encoded = data.copy()
    for col in categorical_cols:
        # convert to str then encode (keeps consistent)
        data_encoded[col] = data_encoded[col].astype(str).fillna("NA")
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        encoder_dict[col] = le

    # numeric columns: ensure numeric dtype (coerce)
    for col in numeric_cols:
        data_encoded[col] = pd.to_numeric(data_encoded[col], errors="coerce").fillna(data_encoded[col].median())

    # -----------------------
    # Scale features and cluster
    # -----------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(data_encoded)

    # Let user choose k or auto use silhouette to pick between 2-6
    st.sidebar.subheader("Clustering settings")
    use_auto_k = st.sidebar.checkbox("Auto-select best k (silhouette)", value=True)
    if use_auto_k:
        best_k = None
        best_score = -1
        for k in range(2, 7):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            try:
                score = silhouette_score(X, labels)
            except:
                score = -1
            if score > best_score:
                best_score = score
                best_k = k
        chosen_k = best_k
        st.sidebar.write(f"Auto-selected k = {chosen_k} (silhouette = {best_score:.3f})")
    else:
        chosen_k = st.sidebar.slider("Choose k (clusters)", 2, 8, 3)

    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    df["Cluster"] = cluster_labels

    # -----------------------
    # Recommendations & descriptions
    # -----------------------
    recommendations = {
        0: [
            "High stress levels detected: try daily 10–15 min guided meditation and breathing exercises.",
            "Improve sleep by removing screens 30–60 minutes before bed and keeping a consistent sleep schedule.",
            "Aim for 30 minutes of moderate exercise most days and maintain hydration."
        ],
        1: [
            "Good work — maintain consistent routines, continue physical activity and social support.",
            "Add occasional journaling or short mindfulness sessions to build resilience.",
            "Monitor screen time and ensure breaks when needed."
        ],
        2: [
            "Reduce continuous screen time: use the Pomodoro technique and take physical breaks.",
            "Introduce outdoor activity 3× per week and limit device multitasking.",
            "Practice short relaxation exercises during high-stress periods."
        ]
    }
    # Generic descriptions — you can adapt / annotate per cluster after EDA
    cluster_descriptions = {
        0: "Cluster 0: Higher stress & poorer sleep habits (interpret via EDA).",
        1: "Cluster 1: Relatively balanced lifestyle, lower stress and consistent sleep.",
        2: "Cluster 2: Moderate issues related to screens and multitasking; mixed exercise habits."
    }

    # -----------------------
    # Sidebar navigation
    # -----------------------
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Insights", "🧾 Cluster Summaries"])

    # ================ HOME PAGE ================
    if page == "🏠 Home":
        st.header("📋 Lifestyle Questionnaire")
        # Numeric sliders (only for columns that exist)
        slider_values = {}
        for col in numeric_cols:
            # default value median
            med = int(data[col].median()) if col in data.columns else 3
            slider_values[col] = st.slider(col + " (1=Low, 5=High)", 1, 5, med)

        # Categorical radios / text input
        # Dynamically get options from original df to avoid missing values
        def opt(col):
            return safe_get_options(df, col)

        # Provide inputs for categorical columns (radio) and allow text for some specific cols
        user_inputs = {}
        for col in categorical_cols:
            # For free-text friendly columns, use text input
            if col.lower().find("activity") >= 0 or col.lower().find("self-care") >= 0 or col.lower().find("self care") >= 0:
                user_inputs[col] = st.text_input(col, "")
            else:
                options = opt(col)
                if options:
                    # use radio to avoid typing
                    user_inputs[col] = st.radio(col, options)
                else:
                    # fallback to text input
                    user_inputs[col] = st.text_input(col, "")

        # Combine numeric & categorical into user dict in same order as data columns
        user_dict = {}
        for c in data.columns:
            if c in slider_values:
                user_dict[c] = slider_values[c]
            elif c in user_inputs:
                val = user_inputs[c]
                # if empty string and original column has categories, fallback to first category
                if val == "" and safe_get_options(df, c):
                    val = safe_get_options(df, c)[0]
                user_dict[c] = val
            else:
                # unexpected column (shouldn't happen) - fill with default
                user_dict[c] = data[c].mode()[0] if not data[c].isna().all() else 0

        if st.button("Get Recommendations"):
            user_df = pd.DataFrame([user_dict])
            # Encode only categorical columns using fitted encoders
            user_encoded = user_df.copy()
            for col in categorical_cols:
                if col in user_encoded.columns:
                    le = encoder_dict.get(col, LabelEncoder())
                    # fit-transform on string representation (if unseen values, transform will raise; handle by mapping)
                    try:
                        user_encoded[col] = le.transform(user_encoded[col].astype(str))
                    except Exception:
                        # unseen label — append to encoder classes and map
                        existing = list(le.classes_)
                        new_val = user_encoded[col].astype(str).iloc[0]
                        classes = existing + [new_val]
                        le.classes_ = classes
                        user_encoded[col] = le.transform(user_encoded[col].astype(str))

            # numeric columns ensure numeric
            for col in numeric_cols:
                if col in user_encoded.columns:
                    user_encoded[col] = pd.to_numeric(user_encoded[col], errors="coerce").fillna(data[col].median())

            # scale and predict
            user_scaled = scaler.transform(user_encoded)
            user_cluster = kmeans.predict(user_scaled)[0]

            # show results + TXT download
            st.subheader(f"📌 You belong to Lifestyle Cluster: {user_cluster}")
            st.markdown(cluster_descriptions.get(user_cluster, "Cluster description not available."))
            st.success("✅ Recommended Actions:")
            rec_text = f"Lifestyle Cluster: {user_cluster}\n"
            rec_text += cluster_descriptions.get(user_cluster, "") + "\n\nRecommendations:\n"
            for r in recommendations.get(user_cluster, []):
                st.write("- " + r)
                rec_text += "- " + r + "\n"

            st.download_button("⬇️ Download Recommendations (TXT)",
                               data=rec_text,
                               file_name="recommendations.txt",
                               mime="text/plain")

    # ================ INSIGHTS PAGE ================
    elif page == "📊 Insights":
        st.header("📊 Cluster Insights & Evaluation")
        # cluster distribution bar
        fig1, ax1 = plt.subplots()
        df["Cluster"].value_counts().sort_index().plot(kind="bar", ax=ax1,
                                                       color=sns.color_palette("Set2", n_colors=chosen_k))
        ax1.set_title("Cluster Distribution")
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        # silhouette
        try:
            sil = silhouette_score(X, cluster_labels)
            st.write(f"Silhouette Score (all features): **{sil:.3f}**")
        except Exception:
            st.write("Silhouette Score: Not available.")

        # average profiles
        st.subheader("Average feature values per cluster (numeric features)")
        cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
        st.dataframe(cluster_profiles)

        # heatmap
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.heatmap(cluster_profiles.T, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
        ax2.set_title("Feature Comparison Heatmap (mean values)")
        st.pyplot(fig2)

        # Allow user to show any feature boxplot by cluster
        st.subheader("Feature distribution by cluster")
        feature = st.selectbox("Select numeric feature to show boxplot", options=list(cluster_profiles.columns))
        fig3, ax3 = plt.subplots(figsize=(8,4))
        sns.boxplot(x="Cluster", y=feature, data=df, ax=ax3, palette="Set2")
        ax3.set_title(f"{feature} by Cluster")
        st.pyplot(fig3)

    # ================ CLUSTER SUMMARIES ================
    elif page == "🧾 Cluster Summaries":
        st.header("🔍 Cluster Descriptions (Plain Language)")
        # Show counts + description for each cluster
        counts = df["Cluster"].value_counts().sort_index()
        for cid in sorted(df["Cluster"].unique()):
            st.markdown(f"### Cluster {cid} — Count: {int(counts.get(cid, 0))}")
            st.markdown(cluster_descriptions.get(cid, "No description available."))
            st.write("Top recommendations:")
            for rec in recommendations.get(cid, []):
                st.write("- " + rec)
            st.markdown("---")

else:
    st.info("⚠️ Please upload your CSV file to begin. Make sure column names match the survey schema used in preprocessing.")
