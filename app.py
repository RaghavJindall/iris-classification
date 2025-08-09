import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🌸 Iris Flower Classification App")
st.write("Predict iris flower species using petal and sepal measurements.")

# Load the trained model
model = joblib.load("model.pkl")

# Create sliders for user input
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.0)

# Prepare feature array for prediction
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(features)[0]
probabilities = model.predict_proba(features)[0]

# Define target names
target_names = ['Setosa', 'Versicolor', 'Virginica']

# Show prediction results
st.subheader("Prediction")
st.write(f"**Species:** {target_names[prediction]}")

st.subheader("Prediction Probabilities")
st.bar_chart(pd.DataFrame(probabilities, index=target_names))
