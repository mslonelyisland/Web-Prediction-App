# milk_test.py
import joblib

# Function to load the model
def load_model():
    model = joblib.load('C:\\Users\\welun\\ITD 105\\Classification\\milk_logistic_model1.pkl')
    return model