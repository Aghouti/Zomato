import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===== Load saved models & scaler =====
xg_model = joblib.load("ModelsAndScaler/XGBoost.pkl")
nn_model = load_model("ModelsAndScaler/neural_network_model.h5")
#scaler = joblib.load("ModelsAndScaler/minmax_scaler.pkl")

# ===== Your feature names (must match training order) =====
feature_names = [
      'name', 'online_order', 'book_table', 'votes',
        'location', 'rest_type',  'cuisines',
       'approx_cost(for two people)', 'reviews_list', 
       'listed_in(type)', 'listed_in(city)'
]

# ===== Title & Sidebar =====
st.title("Restaurant Rating Predictor 🍽️")
model_choice = st.sidebar.selectbox("Select Model", ["XGBoost", "Neural Network"])

# ===== Input Fields =====
st.header("Enter Feature Values")
input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)



# ===== Predict Button =====
if st.button("Predict"):
    if model_choice == "XGBoost":
        prediction = xg_model.predict(feature)
    else:
        prediction = nn_model.predict(feature)
        prediction = np.argmax(prediction, axis=1)  # for classification
    
    # Map numeric predictions to labels
    rating_labels = ["Very Bad", "Bad", "Medium", "Good", "Very Good"]
    pred_label = rating_labels[int(prediction[0])] if int(prediction[0]) < len(rating_labels) else "Unknown"

    st.success(f"Predicted Rating: {pred_label}")

