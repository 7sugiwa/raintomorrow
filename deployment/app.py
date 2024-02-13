# app.py
import streamlit as st
import pandas as pd
import numpy as np
import prediction
import eda  # Import the eda module


# Set up the main structure of the app
def main():
    st.title("Rainfall Prediction in Australia")
    st.write("This app predicts whether it will rain tomorrow in Australia based on weather data.")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Home", "Exploratory Data Analysis", "Make a Prediction"])

    if app_mode == "Home":
        st.write("Welcome to the Weather Forecasting Application!")
        st.write("Navigate to different sections using the sidebar.")
    elif app_mode == "Exploratory Data Analysis":
        st.subheader("Exploratory Data Analysis")
        # Call the EDA function from eda.py
        eda.main()  # Call the main function from eda.py
    elif app_mode == "Make a Prediction":
        st.subheader("Make a Prediction")
        # Get user input for prediction
        user_input = get_user_input()
        if st.button("Predict"):
            # Call the prediction function from prediction.py
            result = prediction.predict_rainfall(*user_input)
            st.write(f"Prediction: {'It will rain tomorrow.' if result else 'No rain tomorrow.'}")

def get_user_input():
    humidity_3pm = st.number_input('Humidity at 3 PM', min_value=0, max_value=100, value=50)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=1000.0, value=0.0)
    rain_today = st.selectbox('Did it rain today?', options=['Yes', 'No'])
    temp_range = st.number_input('Temperature Range (°C)', min_value=0.0, max_value=50.0, value=10.0)
    wind_gust_speed = st.number_input('Wind Gust Speed (km/h)', min_value=0, max_value=100, value=20)
    pressure_9am = st.number_input('Pressure at 9 AM (hPa)', min_value=980, max_value=1040, value=1010)
    avg_pressure = st.number_input('Average Daily Pressure (hPa)', min_value=980, max_value=1040, value=1010)
    humidity_change = st.number_input('Change in Humidity', min_value=-100, max_value=100, value=0)
    avg_humidity = st.number_input('Average Daily Humidity', min_value=0, max_value=100, value=50)
    return humidity_3pm, rainfall, rain_today, temp_range, wind_gust_speed, pressure_9am, avg_pressure, humidity_change, avg_humidity

if __name__ == "__main__":
    main()
