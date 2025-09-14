# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10})

# -----------------------
# Helpers
# -----------------------
def safe_label_encode(le, series):
    vals = series.astype(str).fillna("NA").to_numpy()
    unseen = [v for v in np.unique(vals) if v not in le.classes_]
    if len(unseen) > 0:
        le.classes_ = np.concatenate([le.classes_, np.array(unseen, dtype=le.classes_.dtype)])
    return le.transform(vals)

def safe_get_options(df, col):
    return sorted(df[col].dropna().unique().tolist()) if (col in df.columns and df[col].notnull().any()) else []

def plot_dashboard_clean(df, save_path=None):
    """One clean large dashboard figure (not too congested)."""
    age_col = next((c for c in df.columns if "age" in c.lower()), None)
    gender_col = next((c for c in df.columns if "gender" in c.lower()), None)
    stress_col = next((c for c in df.columns if "stress" in c.lower()), None)
    sleep_col = next((c for c in df.columns if "sleep" in c.lower()), None)
    screen_col = next((c for c in df.columns if "screen" in c.lower()), None)
    breaks_col = next((c for c in df.columns if "break" in c.lower()), None)

    try:
        cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
    except Exception:
        cluster_profiles = None

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 1: Cluster bar
    sns.countplot(x="Cluster", data=df, palette="Set2", order=sorted(df["Cluster"].unique()), ax=axes[0])
    axes[0].set_title("Cluster Distribution")

    # 2: Pie
    counts = df["Cluster"].value_counts().sort_index()
    axes[1].pie(counts, labels=[f"C{int(i)}" for i in counts.index],
                autopct="%1.1f%%", startangle=90,
                colors=sns.color_palette("Set2", n_colors=len(counts)))
    axes[1].set_title("Cluster Share (%)")

    # 3: Age
    if age_col and age_col in df.columns:
        sns.countplot(y=age_col, data=df, order=df[age_col].value_counts().index,
                      palette="Set2", ax=axes[2])
        axes[2].set_title("Age Group Distribution")
    else:
        axes[2].axis("off")

    # 4: Gender
    if gender_col and gender_col in df.columns:
        sns.countplot(x=gender_col, data=df, palette="Set2", ax=axes[3])
        axes[3].set_title("Gender Distribution")
    else:
        axes[3].axis("off")

    # 5: Stress vs sleep
    if stress_col and sleep_col and stress_col in df.columns and sleep_col in df.columns:
        sns.scatterplot(x=stress_col, y=sleep_col, hue="Cluster", data=df,
                        palette="Set2", ax=axes[4], s=50, edgecolor="w")
        axes[4].set_title("Stress vs Sleep Consistency")
    else:
        axes[4].axis("off")

    # 6: Screen vs breaks
    if screen_col and breaks_col and screen_col in df.columns and breaks_col in df.columns:
        sns.scatterplot(x=screen_col, y=breaks_col, hue="Cluster", data=df,
                        palette="Set1", ax=axes[5], s=50, edgecolor="w")
        axes[5].set_title("Screen Time vs Breaks")
    else:
        axes[5].axis("off")

    # 7: Heatmap large
    for ax in axes[6:]:
        ax.axis("off")
    if cluster_profiles is not None and not cluster_profiles.empty:
        sns.heatmap(cluster_profiles.T, annot=True, fmt=".2f", cmap="coolwarm",
                    ax=axes[6], cbar_kws={"shrink":0.6})
        axes[6].set_title("Cluster Feature Means (Heatmap)")

    fig.suptitle("EDA Dashboard — Clustered Lifestyle Insights (Clean Layout)",
                 fontsize=16, y=0.94)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Lifestyle Mental Health Recommender", layout="wide")
st.title("🧠 Lifestyle-Based Mental Health Recommender System")
st.write("Upload your lifestyle survey CSV, explore EDA, and get personalized recommendations (TXT only).")

uploaded_file = st.file_uploader("📂 Upload your Lifestyle & Wellness Survey CSV", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df_raw = pd.read_csv(uploaded_file)
    df_raw.columns = df_raw.columns.str.strip()

    st.sidebar.markdown("**Dataset preview**")
    if st.sidebar.checkbox("Show raw data (first 10 rows)"):
        st.sidebar.dataframe(df_raw.head(10))

    # Drop irrelevant
    data = df_raw.drop(columns=["Timestamp", "Score", "Country"], errors="ignore").copy()

    # Detect numeric
    expected_numeric = [
        "How stressed do you feel due to work?",
        "On a scale of 1–5, how much control do you feel over your time?",
        "How consistent is your sleep schedule?",
        "How often do you take breaks from screen?",
        "How many hours do you sleep on average?",
        "How many hours do you spend on screen daily (excluding work/study)?"
    ]
    numeric_cols = [c for c in expected_numeric if c in data.columns]
    if not numeric_cols:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    categorical_cols = [c for c in data.columns if c not in numeric_cols]

    # Clean categorical
    for col in categorical_cols:
        data[col] = data[col].astype(str).str.strip()
        data[col].replace({"": np.nan, "nan": np.nan, "NA": np.nan,
                           "N/A": np.nan, "None": np.nan}, inplace=True)

    # ==== FIXED: convert numeric-like columns explicitly to numeric BEFORE imputation ====
    if numeric_cols:
        for col in numeric_cols:
            # coerce non-numeric to NaN so imputer can handle them
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Impute
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

    # After imputation check
    if data.isna().sum().sum() > 0:
        st.warning("Warning: Some missing values remain after imputation. Check input file.")
    else:
        st.success("Data cleaned & imputed (no remaining NaNs).")

    # Encode
    encoder_dict = {}
    data_encoded = data.copy()
    for col in categorical_cols:
        data_encoded[col] = data_encoded[col].astype(str).fillna("NA")
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        encoder_dict[col] = le

    for col in numeric_cols:
        data_encoded[col] = pd.to_numeric(data_encoded[col], errors="coerce").fillna(data_encoded[col].median())

    # Scale & cluster
    scaler = StandardScaler()
    X = scaler.fit_transform(data_encoded)

    # ---- Sidebar option: force k=3 or custom ----
    st.sidebar.subheader("Clustering Settings")
    force_k = st.sidebar.checkbox("Force k = 3 (recommended for paper)", value=True)
    if force_k:
        chosen_k = 3
    else:
        chosen_k = st.sidebar.slider("Choose k", min_value=2, max_value=8, value=3)

    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10).fit(X)
    data["Cluster"] = kmeans.labels_
    df = data.copy()

    # Metrics
    try:
        sil = float(silhouette_score(X, kmeans.labels_))
    except Exception:
        sil = None
    try:
        dbi = float(davies_bouldin_score(X, kmeans.labels_))
    except Exception:
        dbi = None

    # Recommendations
    recommendations = {
        0: ["Try daily 10–15 min guided meditation.",
            "Reduce screen exposure before bed.",
            "Aim for 30 min of exercise most days."],
        1: ["Maintain routines & social connections.",
            "Add mindfulness/journaling for resilience.",
            "Monitor screen time & take short breaks."],
        2: ["Use Pomodoro or timed breaks.",
            "Introduce outdoor activity 3× per week.",
            "Practice relaxation exercises during stress."]
    }
    cluster_descriptions = {
        0: "🟢 Cluster 0: Higher stress & poorer sleep habits.",
        1: "🔵 Cluster 1: Balanced lifestyle with healthier routines.",
        2: "🟠 Cluster 2: Moderate issues related to screen-heavy behavior."
    }

    # -----------------------
    # Sidebar navigation
    # -----------------------
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Home", "Insights", "Cluster Summaries", "EDA Dashboard", "Findings & Metrics"])

    # ----------------------- HOME -----------------------
    if page == "Home":
        st.header("📋 Lifestyle Questionnaire")

        slider_values = {}
        for col in numeric_cols:
            default = int(data[col].median()) if col in data.columns else 3
            # ensure slider bounds are valid (1-5) if data are scales; otherwise place safe bounds
            try:
                slider_values[col] = st.slider(col + " (1=Low, 5=High)", 1, 5, int(default), key=f"sl_{col}")
            except Exception:
                # fallback to a general slider
                slider_values[col] = st.slider(col, 0, 100, int(default), key=f"sl_{col}")

        user_inputs = {}
        for col in categorical_cols:
            if "activity" in col.lower() or "self" in col.lower():
                user_inputs[col] = st.text_input(col, key=f"txt_{col}")
            else:
                opts = safe_get_options(data, col)
                if opts:
                    user_inputs[col] = st.radio(col, opts, key=f"rad_{col}")
                else:
                    user_inputs[col] = st.text_input(col, key=f"txtfallback_{col}")

        user_dict = {}
        for c in data.columns:
            if c in slider_values:
                user_dict[c] = slider_values[c]
            elif c in user_inputs:
                val = user_inputs[c]
                if val == "" and safe_get_options(data, c):
                    val = safe_get_options(data, c)[0]
                user_dict[c] = val
            else:
                user_dict[c] = data[c].mode().iat[0] if not data[c].isna().all() else 0

        if st.button("Get Recommendations"):
            user_df = pd.DataFrame([user_dict])

            user_encoded = user_df.copy()
            for col in categorical_cols:
                if col in user_encoded.columns:
                    le = encoder_dict.get(col)
                    if le is not None:
                        user_encoded[col] = safe_label_encode(le, user_encoded[col])
                    else:
                        tmp_le = LabelEncoder()
                        user_encoded[col] = tmp_le.fit_transform(user_encoded[col].astype(str))

            for col in numeric_cols:
                if col in user_encoded.columns:
                    user_encoded[col] = pd.to_numeric(user_encoded[col], errors="coerce").fillna(data[col].median())

            user_scaled = scaler.transform(user_encoded)
            user_cluster = kmeans.predict(user_scaled)[0]

            st.subheader(f"📌 You belong to Lifestyle Cluster: {user_cluster}")
            st.markdown(cluster_descriptions.get(user_cluster, ""))
            st.success("✅ Personalized Recommendations:")
            rec_text = f"Lifestyle Cluster: {user_cluster}\n{cluster_descriptions.get(user_cluster, '')}\n\nRecommendations:\n"
            for r in recommendations.get(user_cluster, []):
                st.write("- " + r)
                rec_text += "- " + r + "\n"

            st.download_button("⬇️ Download Recommendations (TXT)", data=rec_text,
                               file_name="recommendations.txt", mime="text/plain")

    # ----------------------- INSIGHTS -----------------------
    elif page == "Insights":
        st.header("📊 Quick Insights")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Cluster Distribution")
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            df["Cluster"].value_counts().sort_index().plot(kind="bar", ax=ax1, color=sns.color_palette("Set2", n_colors=chosen_k))
            st.pyplot(fig1)
        with col2:
            st.subheader("Cluster % Share")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            df["Cluster"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax2, colors=sns.color_palette("Set2", n_colors=chosen_k))
            ax2.set_ylabel("")
            st.pyplot(fig2)

        if sil is not None:
            if sil < 0.2:
                st.warning(f"Silhouette Score: {sil:.3f} (low — clusters overlap; expected with small/noisy data)")
            else:
                st.info(f"Silhouette Score: {sil:.3f}")
        else:
            st.info("Silhouette Score: Not available")

        if dbi is not None:
            st.write(f"Davies–Bouldin Index: {dbi:.3f}")
        else:
            st.write("Davies–Bouldin Index: Not available")

        cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
        st.subheader("Cluster Profiles (numeric)")
        st.dataframe(cluster_profiles.style.format("{:.2f}"))

    # ----------------------- EDA DASHBOARD -----------------------
    elif page == "EDA Dashboard":
        st.header("📈 EDA Dashboard (Clean Layout)")
        fig = plot_dashboard_clean(df)
        st.pyplot(fig)
        if st.button("📥 Export Dashboard (PNG, 300 dpi)"):
            plot_dashboard_clean(df, save_path="eda_dashboard_paper.png")
            st.success("Saved 'eda_dashboard_paper.png'.")

    # ----------------------- CLUSTER SUMMARIES -----------------------
    elif page == "Cluster Summaries":
        st.header("🧾 Cluster Summaries")
        counts = df["Cluster"].value_counts().sort_index()
        cols = st.columns(3)
        for i, cid in enumerate(range(chosen_k)):
            with cols[i % 3]:
                st.markdown(f"### Cluster {cid}")
                st.write(f"Count: {int(counts.get(cid,0))}")
                st.markdown(cluster_descriptions.get(cid, "No description."))
                st.write("Top recommendations:")
                for rec in recommendations.get(cid, []):
                    st.write("- " + rec)

    # ----------------------- FINDINGS -----------------------
    elif page == "Findings & Metrics":
        st.header("📋 Findings & Metrics")

        findings = [
            ("Cluster Distribution", f"{chosen_k} clusters, imbalance seen", "Clusters 0 & 2 dominant, 1 smaller."),
            ("Age Distribution", "Most respondents 18–28", "Dataset reflects young adults."),
            ("Gender Distribution", "Male slightly higher", "Balanced enough for fair clustering."),
            ("Stress vs Sleep", "High stress → poor sleep", "Cluster 0 most affected."),
            ("Screen vs Breaks", "Cluster 0 heavy screen, few breaks", "Cluster 2 better balance."),
            ("Cluster Profiles", "Distinct averages per cluster", "Stress, sleep & screen habits key factors.")
        ]
        st.table(pd.DataFrame(findings, columns=["Aspect","Observation","Interpretation"]))

        st.subheader("Metrics")
        st.write("Silhouette Score:", f"{sil:.3f}" if sil is not None else "N/A")
        st.write("Davies–Bouldin Index:", f"{dbi:.3f}" if dbi is not None else "N/A")

        limitations = """⚠️ Limitations:
- Small dataset (n≈333) → low silhouette score (≈0.078), clusters overlap.
- Results mostly reflect 18–28 age group, not older populations.
- Clustering exploratory; more data + hybrid models can improve accuracy.
"""
        st.info(limitations)

        # Build findings text safely
        findings_text = "Findings & Metrics\n\n"
        for i, (a,o,interp) in enumerate(findings,1):
            findings_text += f"{i}. {a}\n   - Observation: {o}\n   - Interpretation: {interp}\n\n"
        findings_text += f"Silhouette: {sil:.3f}\nDB Index: {dbi:.3f}\n\n{limitations}" if (sil is not None and dbi is not None) else f"Silhouette: {'N/A' if sil is None else f'{sil:.3f}'}\nDB Index: {'N/A' if dbi is None else f'{dbi:.3f}'}\n\n{limitations}"
        st.download_button("⬇️ Download Findings (TXT)", data=findings_text, file_name="findings.txt", mime="text/plain")

else:
    st.info("⚠️ Please upload your dataset CSV to begin.")
