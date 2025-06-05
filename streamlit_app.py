import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('logistic_model.pkl')

st.title("Logistic Regression Model Predictor")

# Example input fields (customize based on your model's features)
feature1 = st.number_input("Ticket")
feature2 = st.number_input("Sex")
feature3 = st.number_input("Age")
# Add as many as needed

if st.button("Predict"):
    # Convert input to NumPy array
    input_data = np.array([[Ticket, Sex, Age]])  # adjust shape if more features
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")
