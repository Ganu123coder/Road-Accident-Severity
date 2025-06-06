import streamlit as st

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

import pandas as pd
import pickle

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    with open("accident_severity_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# === SIDEBAR NAVIGATION ===
st.sidebar.title("🚦 Go to")
page = st.sidebar.radio("Select Section", ["Home", "Dataset", "Summary", "Predict"])

# =========================
# SECTION: Home
# =========================
if page == "Home":
    st.title("🚨 Road Accident Severity Prediction App")
    st.markdown("""
    This app uses a machine learning model (Random Forest) trained on UK Government Road Safety data to predict **accident severity**.

    **Severity Classes:**
    - 1 = Fatal  
    - 2 = Serious  
    - 3 = Slight  

    Navigate using the sidebar to explore data, view summaries, and generate predictions.
    """)

# =========================
# SECTION: Dataset
# =========================
elif page == "Dataset":
    st.title("📊 Dataset Preview")
    path = "dft-road-casualty-statistics-casualty-2023.csv"
    df = pd.read_csv(path)
    df.drop(columns=["accident_index", "accident_reference", "lsoa_of_casualty"], inplace=True, errors='ignore')
    df.replace(-1, pd.NA, inplace=True)
    df.dropna(inplace=True)
    st.dataframe(df.head(50), use_container_width=True)

# =========================
# SECTION: Summary
# =========================
elif page == "Summary":
    st.title("📈 Dataset Summary")
    path = "/Users/harshuu/Desktop/ML_G/dft-road-casualty-statistics-casualty-2023.csv"
    df = pd.read_csv(path)
    df.drop(columns=["accident_index", "accident_reference", "lsoa_of_casualty"], inplace=True, errors='ignore')
    df.replace(-1, pd.NA, inplace=True)
    df.dropna(inplace=True)
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Severity Value Counts:**")
    st.bar_chart(df["casualty_severity"].value_counts())

# =========================
# SECTION: Predict (User Input)
# =========================
elif page == "Predict":
    st.title("🎯 Predict Accident Severity from Input")

    st.markdown("Enter the required details below:")

    # === INPUT FIELDS ===
    age = st.number_input("Age of Casualty", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex of Casualty", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    casualty_class = st.selectbox("Casualty Class", options=[1, 2, 3], format_func=lambda x: {1: "Driver/Rider", 2: "Passenger", 3: "Pedestrian"}.get(x))
    age_band = st.selectbox("Age Band", options=range(1, 11))
    pedestrian_location = st.selectbox("Pedestrian Location", options=range(0, 7))
    pedestrian_movement = st.selectbox("Pedestrian Movement", options=range(0, 10))
    car_passenger = st.selectbox("Car Passenger", options=range(0, 3))
    bus_passenger = st.selectbox("Bus/Coach Passenger", options=range(0, 2))
    road_worker = st.selectbox("Road Maintenance Worker", options=range(0, 2))
    casualty_type = st.selectbox("Casualty Type", options=range(0, 100))
    home_area_type = st.selectbox("Home Area Type", options=[1, 2])
    imd_decile = st.slider("IMD Decile (1=most deprived, 10=least)", 1, 10)
    enhanced_severity = st.selectbox("Enhanced Severity Code", options=[-1, 0, 1, 2, 3])
    distance_band = st.selectbox("Distance Banding", options=range(1, 6))

    # === PREDICT BUTTON ===
    if st.button("Predict Severity"):
        input_df = pd.DataFrame([{
            'accident_year': 2023,
            'vehicle_reference': 1,
            'casualty_reference': 1,
            'casualty_class': casualty_class,
            'sex_of_casualty': sex,
            'age_of_casualty': age,
            'age_band_of_casualty': age_band,
            'pedestrian_location': pedestrian_location,
            'pedestrian_movement': pedestrian_movement,
            'car_passenger': car_passenger,
            'bus_or_coach_passenger': bus_passenger,
            'pedestrian_road_maintenance_worker': road_worker,
            'casualty_type': casualty_type,
            'casualty_home_area_type': home_area_type,
            'casualty_imd_decile': imd_decile,
            'enhanced_casualty_severity': enhanced_severity,
            'casualty_distance_banding': distance_band
        }])

        prediction = model.predict(input_df)[0]
        severity_label = {1: "Fatal", 2: "Serious", 3: "Slight"}.get(prediction, "Unknown")

        st.success(f"🚦 Predicted Severity: **{severity_label}** (Class {prediction})")
