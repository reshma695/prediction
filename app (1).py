import streamlit as st
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# Load the trained CatBoost model
model = joblib.load("best_catboost_model.pkl")

# Title and description
st.title("Housing Price Prediction App")
st.write("Enter the property details to predict its price.")

# Define the input fields
def user_input_features():
    BuildingArea = st.number_input("Building Area (sq units)", min_value=0.0, step=0.1)
    YearBuilt = st.number_input("Year Built", min_value=1800, step=1)
    PropertyAge = 2024 - YearBuilt  # Calculated feature
    Price_per_SqFt = st.number_input("Price per Square Foot (if known)", min_value=0.0, step=0.1)

    data = {
        "BuildingArea": BuildingArea,
        "YearBuilt": YearBuilt,
        "PropertyAge": PropertyAge,
        "Price_per_SqFt": Price_per_SqFt
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user inputs
st.subheader("User Input Features")
st.write(input_df)

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.subheader("Predicted Price")
    st.write(f"The predicted price of the property is: ${prediction[0]:,.2f}")
