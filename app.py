# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

sns.set_theme()
st.set_page_config(page_title="Lifestyle Recommender (fast start)", layout="wide")

# ---------- Helpers ----------
def safe_get_options(df, col):
    return sorted(df[col].dropna().unique().tolist()) if (col in df.columns and df[col].notnull().any()) else []

def to_numeric_safe(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def impute_safe(df, numeric_cols, categorical_cols):
    if numeric_cols:
        num_imp = SimpleImputer(strategy="median")
        num_filled = num_imp.fit_transform(df[numeric_cols])
        df[numeric_cols] = pd.DataFrame(num_filled, columns=numeric_cols, index=df.index)
    if categorical_cols:
        for c in categorical_cols:
            df[c] = df[c].astype(str)
        cat_imp = SimpleImputer(strategy="most_frequent")
        cat_filled = cat_imp.fit_transform(df[categorical_cols])
        df[categorical_cols] = pd.DataFrame(cat_filled, columns=categorical_cols, index=df.index)
    return df

def quick_plots(df):
    # small safe plots
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    if "Cluster" in df.columns:
        df["Cluster"].value_counts().sort_index().plot(kind="bar", ax=axs[0], color=sns.color_palette("Set2"))
        axs[0].set_title("Cluster Distribution")
    else:
        axs[0].text(0.5,0.5,"No clusters yet", ha="center")
        axs[0].axis("off")

    age_col = next((c for c in df.columns if "age" in c.lower()), None)
    if age_col:
        df[age_col].value_counts().sort_index().plot(kind="bar", ax=axs[1], color=sns.color_palette("Set2"))
        axs[1].set_title("Age distribution")
    else:
        axs[1].text(0.5,0.5,"No age column detected", ha="center")
        axs[1].axis("off")

    plt.tight_layout()
    return fig

# ---------- UI ----------
st.title("🧠 Lifestyle Mental Health Recommender — Fast start")
st.write("This app loads instantly. Upload your CSV and **click** 'Run clustering' to perform heavy computation.")

uploaded = st.file_uploader("Upload your Lifestyle & Wellness Survey CSV", type=["csv"])
if uploaded is None:
    st.info("Please upload your CSV file to begin. (App ready — no heavy work yet.)")
    st.stop()

# Load data (fast)
try:
    df_raw = pd.read_csv(uploaded)
    df_raw.columns = df_raw.columns.str.strip()
except Exception as e:
    st.error("Could not read CSV: " + str(e))
    st.stop()

st.sidebar.subheader("Preview & Controls")
if st.sidebar.checkbox("Show raw preview"):
    st.sidebar.dataframe(df_raw.head(10))

# Detect numeric and categorical heuristically
expected_numeric = [
    "How stressed do you feel due to work?",
    "On a scale of 1–5, how much control do you feel over your time?",
    "How consistent is your sleep schedule?",
    "How often do you take breaks from screen?",
    "How many hours do you sleep on average?",
    "How many hours do you spend on screen daily (excluding work/study)?"
]
numeric_cols = [c for c in expected_numeric if c in df_raw.columns]
if not numeric_cols:
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in df_raw.columns if c not in numeric_cols]

st.sidebar.write(f"Detected numeric cols (sample): {numeric_cols[:6]}")
st.sidebar.write(f"Detected categorical cols (sample): {categorical_cols[:6]}")

# Clean & impute (safe, fast)
df = df_raw.copy()
# normalize categoricals
for c in categorical_cols:
    df[c] = df[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "NA": np.nan, "N/A": np.nan, "None": np.nan})
# convert numeric-like to numeric
df = to_numeric_safe(df, numeric_cols)
# impute
df = impute_safe(df, numeric_cols, categorical_cols)

st.success("Data preprocessed (imputed) — ready for clustering when you are.")

# ---------- Controls: lazy clustering ----------
st.sidebar.subheader("Clustering (lazy)")
force_k3 = st.sidebar.checkbox("Force k = 3", value=True)
if force_k3:
    k = 3
else:
    k = st.sidebar.slider("Choose k", 2, 8, 3)

run = st.sidebar.button("Run clustering now (lazy import)")

# If clustering was already run and cached in session_state, reuse
if "clusters" in st.session_state and st.session_state.get("k_used") == k:
    df["Cluster"] = st.session_state["clusters"]
    st.info(f"Using cached clustering (k={k})")
elif run:
    # do heavy imports only now
    with st.spinner("Importing sklearn and running clustering..."):
        # lazy imports
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        # encode categoricals
        data_enc = df.copy()
        encoders = {}
        for c in categorical_cols:
            le = LabelEncoder()
            data_enc[c] = le.fit_transform(data_enc[c].astype(str))
            encoders[c] = le
        # ensure numeric columns are numeric
        for c in numeric_cols:
            data_enc[c] = pd.to_numeric(data_enc[c], errors="coerce").fillna(df[c].median())

        # scale
        scaler = StandardScaler()
        X = scaler.fit_transform(data_enc)

        # cluster
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        clusters = kmeans.labels_
        df["Cluster"] = clusters

        # metrics
        try:
            sil = silhouette_score(X, clusters)
        except Exception:
            sil = None
        try:
            dbi = davies_bouldin_score(X, clusters)
        except Exception:
            dbi = None

        # cache in session_state
        st.session_state["clusters"] = clusters
        st.session_state["k_used"] = k
        st.session_state["silhouette"] = float(sil) if sil is not None else None
        st.session_state["dbi"] = float(dbi) if dbi is not None else None

        st.success(f"Clustering finished (k={k}). Silhouette: {st.session_state.get('silhouette')}")

# ---------- Show quick EDA & findings ----------
st.header("Quick EDA & Findings")
fig = quick_plots(df)
st.pyplot(fig)

if "clusters" in st.session_state:
    st.subheader("Clustering metrics")
    st.write("Silhouette:", f"{st.session_state['silhouette']:.3f}" if st.session_state.get('silhouette') else "N/A")
    st.write("Davies–Bouldin Index:", f"{st.session_state['dbi']:.3f}" if st.session_state.get('dbi') else "N/A")

st.info("Note: heavy computation and sklearn import only happen when you click 'Run clustering now' (sidebar). This speeds startup on Streamlit Cloud.")
