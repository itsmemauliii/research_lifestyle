import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# ===========================================================================================================================================================================================================
# 1. Load Dataset
# ===========================================================================================================================================================================================================
df = pd.read_csv("Lifestyle & Wellness Insights Survey   (Responses) - Form Responses 1 (2).csv")
df.columns = df.columns.str.strip()
data = df.drop(columns=["Timestamp", "Score", "Country"])

# Encode categorical
categorical_cols = data.select_dtypes(include="object").columns
encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Train KMeans
kmeans_final = KMeans(n_clusters=3, random_state=42)
clusters = kmeans_final.fit_predict(scaled_data)
df["Cluster"] = clusters

# ===========================================================================================================================================================================================================
# 2. Recommendation Rules
# ===========================================================================================================================================================================================================
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

# ===========================================================================================================================================================================================================
# 3. Cluster Descriptions (Plain Language)
# ===========================================================================================================================================================================================================
cluster_descriptions = {
    0: "🟢 **Cluster 0:** High Stress & Poor Sleep — These individuals often feel overworked, use screens before bed, and may lack rest.",
    1: "🔵 **Cluster 1:** Balanced Lifestyle — These users manage stress well, sleep consistently, and engage in healthy routines.",
    2: "🟠 **Cluster 2:** Moderate Issues / Screen-Heavy — These individuals exercise sometimes, but spend long hours on screens and multitasking."
}

# ===========================================================================================================================================================================================================
# 4. Streamlit UI
# ===========================================================================================================================================================================================================
st.set_page_config(page_title="Lifestyle Mental Health Recommender", layout="wide")

st.title("🧠 Lifestyle-Based Mental Health Recommender System")
st.write("Answer the questions below to get personalized mental health recommendations.")

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# User Inputs
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
age = st.selectbox("Age Group:", ["18-28", "29-38", "39-48", "49+"])
gender = st.selectbox("Gender:", ["Male", "Female", "Other"])
stress = st.slider("Stress due to work? (1=Low, 5=High)", 1, 5, 3)
overwhelmed = st.selectbox("How often do you feel overwhelmed?", ["Never", "Rarely", "Sometimes", "Often"])
enjoy_job = st.radio("Do you enjoy your current job/study environment?", ["Yes", "No"])
control_time = st.slider("Control over time (1=Low, 5=High)", 1, 5, 3)
sleep_hours = st.selectbox("Average sleep hours:", ["4-5", "6-7", "8+"])
rested = st.radio("Do you feel rested after waking up?", ["Yes", "No"])
consistent_sleep = st.slider("Consistency of sleep (1=Low, 5=High)", 1, 5, 3)
screens_before_bed = st.radio("Do you use screens before bed?", ["Yes", "No"])
exercise = st.selectbox("Exercise frequency:", ["None", "1-2", "3-5", "Daily"])
physical_activity = st.selectbox("Physical activity:", ["Walk", "Gym", "Sports", "None"])
home_cooked = st.radio("Eat home-cooked meals?", ["Yes", "No"])
water = st.selectbox("Water intake per day:", ["3-4", "5-6", "7-8", "9-10"])
screen_time = st.selectbox("Daily screen time (excluding work/study):", ["<2", "2-4", "5-6", "7+"])
multitask = st.radio("Do you often multitask on devices?", ["Yes", "No"])
breaks = st.slider("Breaks from screen (1=Low, 5=High)", 1, 5, 3)
self_care = st.selectbox("Self-care practices:", ["None", "Meditation", "Journaling", "Other"])

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Predict & Recommend
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

    # Encode user input
    for col in categorical_cols:
        if col in user_df.columns:
            user_df[col] = encoder.fit_transform(user_df[col])

    # Scale
    user_scaled = scaler.transform(user_df)

    # Predict cluster
    cluster = kmeans_final.predict(user_scaled)[0]

    # Show Results
    st.subheader(f"📌 You belong to Lifestyle Cluster: {cluster}")
    st.markdown(cluster_descriptions[cluster])
    st.success("✅ Recommended Actions for You:")
    for rec in recommendations[cluster]:
        st.write("- " + rec)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Cluster Insights Section
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.header("📊 Lifestyle Cluster Insights")

# Cluster distribution
fig, ax = plt.subplots()
df["Cluster"].value_counts().plot(kind="bar", ax=ax, color=["#4CAF50", "#FF9800", "#2196F3"])
ax.set_title("Cluster Distribution")
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Users")
st.pyplot(fig)

# Cluster profiles (numerical features only)
st.subheader("📈 Average Lifestyle Scores per Cluster")
cluster_profiles = df.groupby("Cluster").mean(numeric_only=True)
st.dataframe(cluster_profiles)

# Heatmap for feature comparison
fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(cluster_profiles.T, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
ax.set_title("Cluster Lifestyle Feature Comparison")
st.pyplot(fig)

# Cluster Summaries
st.subheader("🔍 Cluster Descriptions")
for cid, desc in cluster_descriptions.items():
    st.markdown(desc)
