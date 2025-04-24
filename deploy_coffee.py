import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------
# Set page configuration
# --------------------------
st.set_page_config(page_title="Coffee Rating Predictor", layout="centered")

# --------------------------
# Load Models and Encoders
# --------------------------
MODEL_FILES = {
    "rf_model": "coffee_rf_model.pkl",
    "clf_model": "coffee_clf_model.pkl",
    "label_encoders": "coffee_label_encoders.pkl"
}

missing_files = [f for f in MODEL_FILES.values() if not os.path.exists(f)]
if missing_files:
    st.error(f"❌ The following required files are missing: {', '.join(missing_files)}")
    st.stop()

try:
    rf_model = joblib.load(MODEL_FILES["rf_model"])
    clf_model = joblib.load(MODEL_FILES["clf_model"])
    label_encoders = joblib.load(MODEL_FILES["label_encoders"])
except Exception as e:
    st.error(f"⚠️ Error loading model files: {e}")
    st.stop()

# --------------------------
# App Title
# --------------------------
st.title("☕ Coffee Rating & Popularity Predictor")
st.write("Choose the roast, region, and attributes to predict the **customer rating** and **popularity tier** of a coffee product.")

# --------------------------
# User Inputs
# --------------------------
try:
    roast_input = st.selectbox("Select Roast Type", label_encoders['roast types'].classes_)
    region_input = st.selectbox("Select Region", label_encoders['regions'].classes_)
    type_attr_input = st.selectbox("Select Type Attribute", label_encoders['type attributes'].classes_)
except Exception as e:
    st.error(f"❌ Error loading encoder values: {e}")
    st.stop()

# --------------------------
# Encode Inputs
# --------------------------
try:
    roast_encoded = label_encoders['roast types'].transform([roast_input])[0]
    region_encoded = label_encoders['regions'].transform([region_input])[0]
    type_attr_encoded = label_encoders['type attributes'].transform([type_attr_input])[0]

    input_data = pd.DataFrame([[roast_encoded, region_encoded, type_attr_encoded]],
                              columns=['roast types', 'regions', 'type attributes'])
except Exception as e:
    st.error(f"❌ Error encoding the input: {e}")
    st.stop()

# --------------------------
# Prediction
# --------------------------
if st.button("🔮 Predict"):
    try:
        rating_pred = rf_model.predict(input_data)[0]
        tier_pred = clf_model.predict(input_data)[0]

        st.success(f"⭐ **Predicted Rating:** {round(rating_pred, 2)} / 5.0")
        st.info(f"📊 **Popularity Tier:** {tier_pred}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
