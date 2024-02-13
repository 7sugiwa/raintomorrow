# prediction.py
import pickle
import numpy as np
import pandas as pd

# Load the saved model, pipeline, and label encoder
model, pipeline, le = None, None, None

def load_artifacts():
    global model, pipeline, le
    with open('xgboost_optimized_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
    with open('lerain.pkl', 'rb') as file:
        le = pickle.load(file)

load_artifacts()



def predict_rainfall(humidity_3pm, rainfall, rain_today, temp_range, wind_gust_speed, pressure_9am, avg_pressure, humidity_change, avg_humidity):
    # Prepare the feature vector
    data = pd.DataFrame([[humidity_3pm, np.log(rainfall + 1), le.transform([rain_today])[0], temp_range, 
                          wind_gust_speed, pressure_9am, avg_pressure, humidity_change, avg_humidity]],
                        columns=['Humidity3pm', 'Rainfall_log', 'RainToday', 'TempRange', 'WindGustSpeed', 
                                 'Pressure9am', 'AvgPressure', 'HumidityChange', 'AvgHumidity'])
    
    # Apply transformations and make prediction
    transformed_data = pipeline.transform(data)
    prediction = model.predict(transformed_data)
    return prediction[0]
