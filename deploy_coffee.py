import streamlit as st
import pandas as pd
import joblib

# Load models and encoders
rf_model = joblib.load("coffee_rf_model.pkl")
clf_model = joblib.load("coffee_clf_model.pkl")
label_encoders = joblib.load("coffee_label_encoders.pkl")

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
except:
    st.error("❌ One of the selected values could not be encoded. Please check input values.")
    st.stop()

# --------------------------
# Predictions
# --------------------------
if st.button("🔮 Predict"):
    rating_pred = rf_model.predict(input_data)[0]
    tier_pred = clf_model.predict(input_data)[0]

    st.success(f"⭐ **Predicted Rating:** {round(rating_pred, 2)} / 5.0")
    st.info(f"📊 **Popularity Tier:** {tier_pred}")
