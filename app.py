import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load("personality_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🧠 Personality Type Predictor")
st.markdown("This app predicts whether a person is an **Extrovert** or an **Introvert** based on social behavior traits.")

# Input fields
stage_fear = st.selectbox("Stage Fear", ["Yes", "No"])
drained_after_socializing = st.selectbox("Drained After Socializing?", ["Yes", "No"])
time_spent_alone = st.slider("Time Spent Alone (hours/day)", 0, 24, 2)
social_event_attendance = st.slider("Social Event Attendance (per month)", 0, 20, 4)
going_outside = st.slider("Frequency of Going Outside (per week)", 0, 7, 3)
friends_circle_size = st.slider("Friends Circle Size", 0, 50, 10)
post_frequency = st.slider("Social Media Post Frequency (per week)", 0, 20, 5)

# Convert categorical to numeric (manual encoding)
stage_fear_val = 1 if stage_fear == "Yes" else 0
drained_val = 1 if drained_after_socializing == "Yes" else 0

# Combine features
input_data = np.array([[time_spent_alone, stage_fear_val, social_event_attendance,
                        going_outside, drained_val, friends_circle_size, post_frequency]])

# Scale input
scaled_input = scaler.transform(input_data)

# Predict
if st.button("Predict Personality"):
    prediction = model.predict(scaled_input)[0]
    personality = "Introvert" if prediction == 1 else "Extrovert"  # ✅ Correct mapping
    st.success(f"🎯 Predicted Personality: **{personality}**")
