import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler # We need this definition, even if not fitting

# --- Load Model and Scaler ---
# Ensure these file paths are correct relative to where app.py is
try:
    model = joblib.load('iris_model.joblib')
    scaler = joblib.load('iris_scaler.joblib')
except FileNotFoundError:
    st.error("Model or scaler file not found. Make sure 'iris_model.joblib' and 'iris_scaler.joblib' are in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()


# --- Define Iris feature names and target names (for clarity) ---
# These should match the original dataset's features
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
target_names = ['setosa', 'versicolor', 'virginica'] # From iris.target_names

# --- Streamlit App Interface ---
st.set_page_config(page_title="Iris Flower Classifier", layout="wide")
st.title("🌸 Iris Flower Species Classifier")
st.markdown("""
This app predicts the species of an Iris flower based on its sepal and petal measurements.
Please provide the measurements below.
""")

# --- Input fields in the sidebar ---
st.sidebar.header("Input Flower Measurements (cm)")

input_features = {}
# Create sliders for each feature
# Use reasonable min/max values based on the Iris dataset's typical ranges
# From df.describe() in your Colab notebook:
# sepal length (cm): min=4.3, max=7.9
# sepal width (cm):  min=2.0, max=4.4
# petal length (cm): min=1.0, max=6.9
# petal width (cm):  min=0.1, max=2.5

default_values = {
    'sepal length (cm)': 5.8, # Mean of sepal length
    'sepal width (cm)': 3.0,  # Mean of sepal width
    'petal length (cm)': 3.7, # Mean of petal length
    'petal width (cm)': 1.2   # Mean of petal width
}

min_max_values = {
    'sepal length (cm)': (4.0, 8.0),
    'sepal width (cm)': (2.0, 4.5),
    'petal length (cm)': (1.0, 7.0),
    'petal width (cm)': (0.1, 2.5)
}

for feature in feature_names:
    input_features[feature] = st.sidebar.slider(
        label=feature.capitalize(),
        min_value=min_max_values[feature][0],
        max_value=min_max_values[feature][1],
        value=default_values[feature], # Default to average values
        step=0.1
    )

# Convert input features to a DataFrame (or NumPy array in the correct order)
input_df = pd.DataFrame([input_features])

# --- Prediction Logic ---
if st.sidebar.button("Predict Species"):
    if model and scaler:
        # Ensure the order of columns in input_df matches the training data
        input_data_ordered = input_df[feature_names].values

        # Scale the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data_ordered)

        # Make prediction
        prediction_index = model.predict(input_data_scaled)[0]
        prediction_name = target_names[prediction_index]

        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_data_scaled)[0]

        # --- Display Results ---
        st.subheader("Prediction:")
        st.success(f"The predicted species is: **{prediction_name.capitalize()}**")

        st.subheader("Prediction Probabilities:")
        # Create a DataFrame for better display of probabilities
        proba_df = pd.DataFrame({
            'Species': target_names,
            'Probability': prediction_proba
        })
        proba_df['Probability'] = proba_df['Probability'].apply(lambda x: f"{x:.2%}") # Format as percentage
        st.table(proba_df.set_index('Species'))

        # Optional: Display input features
        st.subheader("Input Features:")
        st.table(input_df.rename(columns=lambda x: x.capitalize()))

    else:
        st.error("Model or scaler not loaded. Cannot make prediction.")
else:
    st.info("Adjust the sliders in the sidebar and click 'Predict Species'.")

st.markdown("---")
st.markdown("Built with Streamlit by [Your Name/Handle]")
