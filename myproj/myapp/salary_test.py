import joblib
import numpy as np

# Load the trained model, label encoders, and scaler
model = joblib.load('C:\\Users\\welun\\ITD 105\\Classification\\logistic_model.pkl')
encoders = joblib.load('C:\\Users\\welun\\ITD 105\\Classification\\label_encoders.pkl')
scaler = joblib.load('C:\\Users\\welun\\ITD 105\\Classification\\scaler.pkl')

def predict_danger(animal_name, symptoms):
    # Prepare the input data
    input_data = [animal_name] + symptoms
    encoded_data = []

    for col, val in zip(encoders.keys(), input_data):
        if val in encoders[col].classes_:
            encoded_val = encoders[col].transform([val])[0]
        else:
            encoded_val = encoders[col].transform([encoders[col].classes_[0]])[0]
        encoded_data.append(encoded_val)

    # Ensure the data is in the correct shape for scaling
    encoded_array = np.array(encoded_data).reshape(1, -1)

    # Scale the features
    scaled_data = scaler.transform(encoded_array)

    # Predict
    prediction = model.predict(scaled_data)
    return encoders['Dangerous'].inverse_transform(prediction)[0]

