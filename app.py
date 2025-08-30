import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# =======================
# 1. Streamlit Setup
# =======================
st.set_page_config(page_title="Lifestyle Mental Health Recommender", layout="wide")

st.title("🧠 Lifestyle-Based Mental Health Recommender System")
st.write("Upload your survey dataset and answer the questions below to get personalized recommendations.")

# =======================
# 2. File Upload
# =======================
uploaded_file = st.file_uploader("📂 Upload your Lifestyle & Wellness Survey CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Drop irrelevant columns
    data = df.drop(columns=["Timestamp", "Score", "Country"], errors="ignore")

    # =======================
    # 3. Explicit numeric vs categorical
    # =======================
    numeric_cols = [
        "How stressed do you feel due to work?",
        "On a scale of 1–5, how much control do you feel over your time?",
        "How consistent is your sleep schedule?",
        "How often do you take breaks from screen?"
    ]
    categorical_cols = [col for col in data.columns if col not in numeric_cols]

    # Encode categoricals
    encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col].astype(str))

    # Scale
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # KMeans
    kmeans_final = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans_final.fit_predict(scaled_data)
    df["Cluster"] = clusters

    # =======================
    # 4. Rules
    # =======================
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
        0: "🟢 **Cluster 0:** High Stress & Poor Sleep — Often feel overworked, use screens before bed, may lack rest.",
        1: "🔵 **Cluster 1:** Balanced Lifestyle — Manage stress well, sleep consistently, engage in healthy routines.",
        2: "🟠 **Cluster 2:** Moderate Issues / Screen-Heavy — Exercise sometimes, but spend long hours on screens and multitasking."
    }

    # =======================
    # 5. Sidebar Navigation
    # =======================
    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 Insights"])

    # =======================
    # 6. Home Page
    # =======================
    if page == "🏠 Home":
        st.header("📋 Lifestyle Questionnaire")

        # Numeric sliders
        stress = st.slider("Stress due to work? (1=Low, 5=High)", 1, 5, 3)
        control_time = st.slider("Control over time (1=Low, 5=High)", 1, 5, 3)
        consistent_sleep = st.slider("Consistency of sleep (1=Low, 5=High)", 1, 5, 3)
        breaks = st.slider("Breaks from screen (1=Low, 5=High)", 1, 5, 3)

        # Extract options safely
        def get_options(col):
            return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

        # Radios (categoricals)
        age = st.radio("Age Group:", get_options("Age"))
        gender = st.radio("Gender:", get_options("Gender"))
        overwhelmed = st.radio("How often do you feel overwhelmed?", get_options("How often do you feel overwhelmed in a week?"))
        enjoy_job = st.radio("Do you enjoy your current job/study environment?", get_options("Do you enjoy your current job/study environment?"))
        sleep_hours = st.radio("Average sleep hours:", get_options("How many hours do you sleep on average?"))
        rested = st.radio("Do you feel rested after waking up?", get_options("Do you feel rested after waking up?"))
        screens_before_bed = st.radio("Do you use screens before bed?", get_options("Do you use screens before bed?"))
        exercise = st.radio("Exercise frequency:", get_options("How often do you exercise in a week?"))
        home_cooked = st.radio("Eat home-cooked meals?", get_options("Do you eat home-cooked meals most of the time?"))
        water = st.radio("Water intake per day:", get_options("How many glasses of water do you drink per day?"))
        screen_time = st.radio("Daily screen time (excluding work/study):", get_options("How many hours do you spend on screen daily (excluding work/study)?"))
        multitask = st.radio("Do you often multitask on devices?", get_options("Do you often multitask on devices?"))

        # Free text inputs
        physical_activity = st.text_input("What kind of physical activity do you engage in? (e.g., Walk, Yoga, Cycling)")
        self_care = st.text_input("Do you follow any self-care practices? (e.g., Meditation, Journaling, None)")

        # =======================
        # 7. Prediction + TXT Download
        # =======================
        if st.button("Get Recommendations"):
            user_dict = {
                "Age": [age],
                "Gender": [gender],
                "How stressed do you feel due to work?": [stress],
                "How often do you feel overwhelmed in a week?": [overwhelmed],
                "Do you enjoy your current job/study environment?": [enjoy_job],
                "On a scale of 1–5, how much control do you feel over your time?": [control_time],
                "How many hours do you sleep on average?": [sleep_hours],
                "Do you feel rested after waking up?": [rested],
                "How consistent is your sleep schedule?": [consistent_sleep],
                "Do you use screens before bed?": [screens_before_bed],
                "How often do you exercise in a week?": [exercise],
                "What kind of physical activity do you engage in?": [physical_activity],
                "Do you eat home-cooked meals most of the time?": [home_cooked],
                "How many glasses of water do you drink per day?": [water],
                "How many hours do you spend on screen daily (excluding work/study)?": [screen_time],
                "Do you often multitask on devices?": [multitask],
                "How often do you take breaks from screen?": [breaks],
                "Do you follow any self-care practices?": [self_care]
            }

            user_df = pd.DataFrame(user_dict)

            # Encode categoricals only
            for col in categorical_cols:
                if col in user_df.columns:
                    user_df[col] = encoder.fit_transform(user_df[col].astype(str))

            # Scale
            user_scaled = scaler.transform(user_df)

            # Predict cluster
            cluster = kmeans_final.predict(user_scaled)[0]

            st.subheader(f"📌 You belong to Lifestyle Cluster: {cluster}")
            st.markdown(cluster_descriptions[cluster])
            st.success("✅ Recommended Actions for You:")

            rec_text = f"📌 Lifestyle Cluster: {cluster}\n"
            rec_text += cluster_descriptions[cluster] + "\n\n"
            rec_text += "✅ Recommendations:\n"
            for rec in recommendations[cluster]:
                st.write("- " + rec)
                rec_text += "- " + rec + "\n"

            # TXT Download only
            st.download_button(
                label="⬇️ Download Recommendations (TXT)",
                data=rec_text,
                file_name="recommendations.txt",
                mime="text/plain"
            )

    # =======================
    # 8. Insights Page
    # =======================
    elif page == "📊 Insights":
        st.header("📊 Lifestyle Cluster Insights")

        # Cluster distribution
        st.subheader("📊 Cluster Distribution")
        fig, ax = plt.subplots()
        df["Cluster"].value_counts().plot(kind="bar", ax=ax, color=["#4CAF50", "#FF9800", "#2196F3"])
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of Users")
        st.pyplot(fig)

        # Cluster profiles
        st.subheader("📈 Average Lifestyle Scores per Cluster")
        cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
        st.dataframe(cluster_profiles)

        # Heatmap
        st.subheader("🔥 Feature Comparison (Heatmap)")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(cluster_profiles.T, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Descriptions
        st.subheader("🔍 Cluster Descriptions")
        for cid, desc in cluster_descriptions.items():
            st.markdown(desc)

else:
    st.warning("⚠️ Please upload a dataset CSV to continue.")


