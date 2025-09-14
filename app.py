# app.py (lazy sklearn imports + robust impute + caching)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

sns.set_theme()
st.set_page_config(page_title="Lifestyle Mental Health Recommender", layout="wide")

# -----------------------
# Helpers
# -----------------------
def safe_get_options(df, col):
    return sorted(df[col].dropna().unique().tolist()) if (col in df.columns and df[col].notnull().any()) else []

def safe_label_encode(le, series):
    vals = series.astype(str).fillna("NA").to_numpy()
    unseen = [v for v in np.unique(vals) if v not in le.classes_]
    if len(unseen) > 0:
        le.classes_ = np.concatenate([le.classes_, np.array(unseen, dtype=le.classes_.dtype)])
    return le.transform(vals)

# -----------------------
# Caching data load & preprocess
# -----------------------
@st.cache_data
def load_df(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def preprocess_df(df):
    # drop irrelevant
    data = df.drop(columns=["Timestamp", "Score", "Country"], errors="ignore").copy()

    # expected numeric columns (adjust if your CSV has different names)
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
        # fallback: treat existing numeric dtypes as numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    categorical_cols = [c for c in data.columns if c not in numeric_cols]

    # normalize categorical values
    for c in categorical_cols:
        data[c] = data[c].astype(str).str.strip()
        data[c].replace({"": np.nan, "nan": np.nan, "NA": np.nan, "N/A": np.nan, "None": np.nan}, inplace=True)

    # convert numeric-like to numeric
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # impute numeric -> median (wrap into DataFrame to keep index/cols)
    if numeric_cols:
        num_imp = SimpleImputer(strategy="median")
        num_filled = num_imp.fit_transform(data[numeric_cols])
        data_num_df = pd.DataFrame(num_filled, columns=numeric_cols, index=data.index)
        data[numeric_cols] = data_num_df

    # impute categorical -> most frequent
    if categorical_cols:
        for c in categorical_cols:
            data[c] = data[c].astype(str)
        cat_imp = SimpleImputer(strategy="most_frequent")
        cat_filled = cat_imp.fit_transform(data[categorical_cols])
        data_cat_df = pd.DataFrame(cat_filled, columns=categorical_cols, index=data.index)
        data[categorical_cols] = data_cat_df

    return data, numeric_cols, categorical_cols

@st.cache_resource
def build_encoders_and_scaler(data, categorical_cols):
    # build label encoders for categorical cols
    encoders = {}
    data_encoded = data.copy()
    for c in categorical_cols:
        from sklearn.preprocessing import LabelEncoder  # local import ok
        le = LabelEncoder()
        data_encoded[c] = le.fit_transform(data_encoded[c])
        encoders[c] = le
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(data_encoded)
    return encoders, scaler, X, data_encoded

# -----------------------
# Plot helpers
# -----------------------
def plot_dashboard_clean(df):
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

    fig, axes = plt.subplots(3,3, figsize=(18,12))
    axes = axes.flatten()
    sns.countplot(x="Cluster", data=df, palette="Set2", order=sorted(df["Cluster"].unique()), ax=axes[0])
    axes[0].set_title("Cluster Distribution")

    counts = df["Cluster"].value_counts().sort_index()
    axes[1].pie(counts, labels=[f"C{int(i)}" for i in counts.index], autopct="%1.1f%%", startangle=90, colors=sns.color_palette("Set2", n_colors=len(counts)))
    axes[1].set_title("Cluster Share (%)")

    if age_col and age_col in df.columns:
        sns.countplot(y=age_col, data=df, order=df[age_col].value_counts().index, palette="Set2", ax=axes[2])
        axes[2].set_title("Age Group Distribution")
    else:
        axes[2].axis("off")

    if gender_col and gender_col in df.columns:
        sns.countplot(x=gender_col, data=df, palette="Set2", ax=axes[3])
        axes[3].set_title("Gender Distribution")
    else:
        axes[3].axis("off")

    if stress_col and sleep_col and stress_col in df.columns and sleep_col in df.columns:
        sns.scatterplot(x=stress_col, y=sleep_col, hue="Cluster", data=df, palette="Set2", ax=axes[4], s=50, edgecolor="w")
        axes[4].set_title("Stress vs Sleep Consistency")
    else:
        axes[4].axis("off")

    if screen_col and breaks_col and screen_col in df.columns and breaks_col in df.columns:
        sns.scatterplot(x=screen_col, y=breaks_col, hue="Cluster", data=df, palette="Set1", ax=axes[5], s=50, edgecolor="w")
        axes[5].set_title("Screen Time vs Breaks")
    else:
        axes[5].axis("off")

    for ax in axes[6:]:
        ax.axis("off")
    if cluster_profiles is not None and not cluster_profiles.empty:
        sns.heatmap(cluster_profiles.T, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[6], cbar_kws={"shrink":0.6})
        axes[6].set_title("Cluster Feature Means (Heatmap)")

    fig.suptitle("EDA Dashboard — Clustered Lifestyle Insights (Clean Layout)", fontsize=16, y=0.94)
    plt.tight_layout(rect=[0,0,1,0.93])
    return fig

# -----------------------
# UI & Flow
# -----------------------
st.title("🧠 Lifestyle-Based Mental Health Recommender System")
uploaded = st.file_uploader("Upload your Lifestyle & Wellness Survey CSV", type=["csv"])

if uploaded is None:
    st.info("Upload your dataset (CSV) to begin.")
    st.stop()

# load & preprocess
try:
    raw_df = load_df(uploaded)
    data, numeric_cols, categorical_cols = preprocess_df(raw_df)
except Exception as e:
    st.error("Failed loading/preprocessing: " + str(e))
    st.stop()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Insights", "Cluster Summaries", "EDA Dashboard", "Findings & Metrics"])

# prepare encoders/scaler but don't import heavy sklearn clustering yet
try:
    encoders, scaler, X, data_encoded = build_encoders_and_scaler(data, categorical_cols)
except Exception as e:
    st.error("Encoder/scaler build failed: " + str(e))
    st.stop()

# CLUSTERING controls (deferred)
st.sidebar.subheader("Clustering Controls")
force_k = st.sidebar.checkbox("Force k = 3 (recommended for paper)", value=True)
if force_k:
    chosen_k = 3
else:
    chosen_k = st.sidebar.slider("Choose number of clusters (k)", 2, 8, 3)

# If clustering already run and stored in session_state, reuse
if "kmeans" in st.session_state and "clusters" in st.session_state:
    kmeans = st.session_state["kmeans"]
    data["Cluster"] = st.session_state["clusters"]
else:
    st.sidebar.markdown("Click to run clustering (lazy import, runs once)")
    if st.sidebar.button("Run clustering now"):
        with st.spinner("Running clustering (this may take a short while)..."):
            # lazy imports for sklearn
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score, davies_bouldin_score

            kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
            kmeans.fit(X)
            clusters = kmeans.labels_
            data["Cluster"] = clusters
            # store to session_state
            st.session_state["kmeans"] = kmeans
            st.session_state["clusters"] = clusters
            # compute metrics and store
            try:
                st.session_state["silhouette"] = float(silhouette_score(X, clusters))
            except Exception:
                st.session_state["silhouette"] = None
            try:
                st.session_state["dbi"] = float(davies_bouldin_score(X, clusters))
            except Exception:
                st.session_state["dbi"] = None
            st.success("Clustering completed and cached.")

# Pages ----------------------------------------------------------------
# HOME - user input and recommendations (encoders available)
if page == "Home":
    st.header("📋 Lifestyle Questionnaire")
    slider_values = {}
    for col in numeric_cols:
        default = int(data[col].median()) if col in data.columns else 3
        try:
            slider_values[col] = st.slider(col + " (1=Low, 5=High)", 1, 5, default, key=f"sl_{col}")
        except Exception:
            slider_values[col] = st.number_input(col, value=int(default), key=f"num_{col}")

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
            v = user_inputs[c]
            if v == "" and safe_get_options(data, c):
                v = safe_get_options(data, c)[0]
            user_dict[c] = v
        else:
            user_dict[c] = data[c].mode().iat[0] if not data[c].isna().all() else 0

    if st.button("Get Recommendations for my inputs"):
        if "kmeans" not in st.session_state:
            st.error("Please run clustering first (sidebar → Run clustering now).")
        else:
            # encode user inputs
            user_df = pd.DataFrame([user_dict])
            user_enc = user_df.copy()
            for col in categorical_cols:
                if col in encoders:
                    user_enc[col] = safe_label_encode(encoders[col], user_enc[col])
            for col in numeric_cols:
                if col in user_enc.columns:
                    user_enc[col] = pd.to_numeric(user_enc[col], errors="coerce").fillna(data[col].median())
            user_scaled = scaler.transform(user_enc)
            user_cluster = st.session_state["kmeans"].predict(user_scaled)[0]

            descriptions = {
                0: "🟢 Cluster 0: Higher stress & poorer sleep habits.",
                1: "🔵 Cluster 1: Balanced lifestyle with healthier routines.",
                2: "🟠 Cluster 2: Moderate issues related to screen-heavy behavior."
            }
            recs = {
                0: ["Try daily 10–15 min guided meditation.", "Reduce screen exposure before bed.", "Aim for 30 min exercise most days."],
                1: ["Keep routines & social connections.", "Add mindfulness/journaling.", "Monitor screen time & take breaks."],
                2: ["Use Pomodoro breaks.", "Introduce outdoor activity 3×/week.", "Practice relaxation exercises."]
            }

            st.subheader(f"Your predicted cluster: {user_cluster}")
            st.markdown(descriptions.get(user_cluster, "Cluster description not found."))
            st.write("Recommendations:")
            for r in recs.get(user_cluster, []):
                st.write("- " + r)
            rec_text = f"Cluster: {user_cluster}\n" + "\n".join(recs.get(user_cluster, []))
            st.download_button("Download Recommendations (TXT)", data=rec_text, file_name="recommendations.txt", mime="text/plain")

# INSIGHTS
elif page == "Insights":
    st.header("📊 Insights")
    if "kmeans" not in st.session_state:
        st.info("Clustering not yet run. Run clustering from sidebar to view cluster insights.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(4,3))
            data["Cluster"].value_counts().sort_index().plot(kind="bar", ax=ax1, color=sns.color_palette("Set2", n_colors=chosen_k))
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(4,3))
            data["Cluster"].value_counts().plot(kind="pie", autopct="%1.1f%%", startangle=90, ax=ax2, colors=sns.color_palette("Set2", n_colors=chosen_k))
            ax2.set_ylabel("")
            st.pyplot(fig2)

        sil = st.session_state.get("silhouette", None)
        dbi = st.session_state.get("dbi", None)
        st.write("Silhouette:", f"{sil:.3f}" if sil is not None else "N/A")
        st.write("Davies-Bouldin Index:", f"{dbi:.3f}" if dbi is not None else "N/A")

        cluster_profiles = data.groupby("Cluster").mean(numeric_only=True)
        st.subheader("Cluster numeric means")
        st.dataframe(cluster_profiles.style.format("{:.2f}"))

# CLUSTER SUMMARIES
elif page == "Cluster Summaries":
    st.header("🧾 Cluster Summaries")
    if "kmeans" not in st.session_state:
        st.info("Run clustering first.")
    else:
        counts = data["Cluster"].value_counts().sort_index()
        cols = st.columns(3)
        for i, cid in enumerate(range(chosen_k)):
            with cols[i % 3]:
                st.markdown(f"### Cluster {cid}")
                st.write(f"Count: {int(counts.get(cid,0))}")
                # simple static descriptions
                desc = "No description."
                if cid == 0: desc = "Higher stress & poorer sleep habits."
                if cid == 1: desc = "Balanced lifestyle."
                if cid == 2: desc = "Moderate screen-heavy behavior."
                st.write(desc)
