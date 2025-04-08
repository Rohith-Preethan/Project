import streamlit as st
import pandas as pd
import pickle

# Load trained model
try:
    with open("sky_server.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ trained_model.pkl not found. Please train and save the model first.")
    st.stop()

st.set_page_config(page_title="SkyServer Object Classifier")
st.title("🛰️ SkyServer Object Classifier")
st.write("Enter feature values to predict the type of celestial object.")

# List of input features
feature_names = [
    'ra', 'dec', 'u', 'g', 'r', 'i', 'z',
    'run', 'camcol', 'field', 'redshift',
    'plate', 'mjd', 'fiberid',
    'u_g', 'g_r'
]

user_input = {}

# Create input fields
with st.form("prediction_form"):
    for feature in feature_names:
        user_input[feature] = st.number_input(f"{feature}", format="%.4f")
    
    submitted = st.form_submit_button("🔮 Predict")

# On form submission
if submitted:
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)
    st.success(f"🧬 Predicted Class: {prediction[0]}")
