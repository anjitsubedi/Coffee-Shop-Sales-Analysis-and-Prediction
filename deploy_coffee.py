import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------
# Debug info for paths
# --------------------------
st.write("📂 Current Working Directory:", os.getcwd())
st.write("📁 Files in Directory:", os.listdir())

# --------------------------
# Load models and encoders safely
# --------------------------
try:
    rf_model = joblib.load("coffee_rf_model.pkl")
    clf_model = joblib.load("coffee_clf_model.pkl")
    label_encoders = joblib.load("coffee_label_encoders.pkl")
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Unexpected error loading models: {e}")
    st.stop()

# --------------------------
# App Config
# --------------------------
st.set_page_config(page_title="Coffee Rating Predictor", layout="centered")

# --------------------------
# App Title
# --------------------------
st.title("☕ Coffee Rating & Popularity Predictor")
st.write("Choose the roast, region, and attributes to predict the **customer rating** and **popularity tier** of a coffee product.")

# --------------------------
# User Inputs
# --------------------------
roast_input = st.selectbox("Select Roast Type", label_encoders['roast types'].classes_)
region_input = st.selectbox("Select Region", label_encoders['regions'].classes_)
type_attr_input = st.selectbox("Select Type Attribute", label_encoders['type attributes'].classes_)

# --------------------------
# Encoding Inputs
# --------------------------
try:
    roast_encoded = label_encoders['roast types'].transform([roast_input])[0]
    region_encoded = label_encoders['regions'].transform([region_input])[0]
    type_attr_encoded = label_encoders['type attributes'].transform([type_attr_input])[0]

    input_data = pd.DataFrame([[roast_encoded, region_encoded, type_attr_encoded]],
                              columns=['roast types', 'regions', 'type attributes'])
except Exception as e:
    st.error(f"❌ Error encoding input values: {e}")
    st.stop()

# --------------------------
# Predictions
# --------------------------
if st.button("🔮 Predict"):
    try:
        rating_pred = rf_model.predict(input_data)[0]
        tier_pred = clf_model.predict(input_data)[0]

        st.success(f"⭐ **Predicted Rating:** {round(rating_pred, 2)} / 5.0")
        st.info(f"📊 **Popularity Tier:** {tier_pred}")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
