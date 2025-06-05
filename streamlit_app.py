import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('logistic_model.pkl')

st.title("Logistic Regression Model Predictor")

# Example input fields (customize based on your model's features)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
# Add as many as needed

if st.button("Predict"):
    # Convert input to NumPy array
    input_data = np.array([[feature1, feature2, feature3]])  # adjust shape if more features
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")
