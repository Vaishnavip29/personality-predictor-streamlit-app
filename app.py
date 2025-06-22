import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Personality Predictor", layout="centered")
st.title("🧠 Personality Type Predictor")

# ⏳ Train the model only once
@st.cache_data
def train_model():
    # Load your dataset here if not using synthetic data:
    # df = pd.read_csv("personality_dataset.csv")

    # For demonstration, we'll use a synthetic balanced dataset
    np.random.seed(42)
    n = 100
    extroverts = pd.DataFrame({
        "Time_spent_Alone": np.random.randint(0, 5, n),
        "Stage_fear": np.random.choice(["No", "Yes"], n, p=[0.8, 0.2]),
        "Social_event_attendance": np.random.randint(5, 20, n),
        "Going_outside": np.random.randint(4, 7, n),
        "Drained_after_socializing": np.random.choice(["No", "Yes"], n, p=[0.8, 0.2]),
        "Friends_circle_size": np.random.randint(10, 40, n),
        "Post_frequency": np.random.randint(5, 20, n),
        "Personality": "Extrovert"
    })
    introverts = pd.DataFrame({
        "Time_spent_Alone": np.random.randint(5, 15, n),
        "Stage_fear": np.random.choice(["No", "Yes"], n, p=[0.2, 0.8]),
        "Social_event_attendance": np.random.randint(0, 6, n),
        "Going_outside": np.random.randint(0, 4, n),
        "Drained_after_socializing": np.random.choice(["No", "Yes"], n, p=[0.2, 0.8]),
        "Friends_circle_size": np.random.randint(0, 15, n),
        "Post_frequency": np.random.randint(0, 5, n),
        "Personality": "Introvert"
    })
    df = pd.concat([extroverts, introverts], ignore_index=True).sample(frac=1).reset_index(drop=True)

    # Encode categorical features
    df['Stage_fear'] = LabelEncoder().fit_transform(df['Stage_fear'])  # Yes=1
    df['Drained_after_socializing'] = LabelEncoder().fit_transform(df['Drained_after_socializing'])
    df['Personality'] = df['Personality'].map({'Extrovert': 0, 'Introvert': 1})

    # Features and target
    X = df.drop("Personality", axis=1)
    y = df["Personality"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split and model training
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, scaler

# 🔁 Train and cache model
model, scaler = train_model()

# 🎯 Input form
st.subheader("📝 Enter Social Behavior Traits")

time_spent_alone = st.slider("Time Spent Alone (hours/day)", 0, 24, 2)
stage_fear = st.radio("Do you have stage fear?", ["Yes", "No"])
social_event_attendance = st.slider("Social Events per Month", 0, 20, 5)
going_outside = st.slider("How often do you go outside per week?", 0, 7, 3)
drained_after_socializing = st.radio("Feel drained after socializing?", ["Yes", "No"])
friends_circle_size = st.slider("Number of close friends", 0, 50, 10)
post_frequency = st.slider("How often do you post on social media (per week)?", 0, 20, 5)

# 🧮 Prepare input
stage_val = 1 if stage_fear == "Yes" else 0
drain_val = 1 if drained_after_socializing == "Yes" else 0

input_data = np.array([[time_spent_alone, stage_val, social_event_attendance,
                        going_outside, drain_val, friends_circle_size, post_frequency]])
scaled_input = scaler.transform(input_data)

# 🧠 Predict
if st.button("Predict Personality"):
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0]

    personality = "Introvert 🧍‍♂️" if prediction == 1 else "Extrovert 🧑‍🤝‍🧑"
    st.success(f"🎯 Predicted Personality: **{personality}**")

    st.markdown("### 🔍 Confidence Scores")
    st.write(f"Introvert: `{proba[1]:.2f}`")
    st.write(f"Extrovert: `{proba[0]:.2f}`")

    st.caption("Note: 0 = Extrovert, 1 = Introvert in the model.")
