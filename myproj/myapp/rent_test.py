from django.shortcuts import render
import joblib
import pandas as pd  # Make sure to import pandas

# Load your trained model
model = joblib.load('C:\\Users\\welun\\ITD 105\\Regression\\rent_prediction_model.pkl')

def predict_rent(request):
    if request.method == 'POST':
        # Extract features from form data and convert them to the correct data type
        BHK = int(request.POST.get('BHK'))
        Size = float(request.POST.get('Size'))
        Floor = int(request.POST.get('Floor'))
        Bathroom = int(request.POST.get('Bathroom'))
        FurnishingStatus = request.POST.get('FurnishingStatus')
        AreaType = request.POST.get('AreaType')
        AreaLocality = request.POST.get('AreaLocality')
        City = request.POST.get('City')
        # TenantPreferred = request.POST.get('TenantPreferred')

        # Prepare the feature vector as a DataFrame for prediction
        features_df = pd.DataFrame({
            'BHK': [BHK],
            'Size': [Size],
            'Floor': [Floor],
            'Bathroom': [Bathroom],
            'Furnishing Status': [FurnishingStatus],
            'Area Type' : [AreaType],
            'Area Locality' : [AreaLocality],
            'City' : [City],
            # 'Tenant Preferred': [TenantPreferred]
        })

        # Predict using the loaded model
        prediction = model.predict(features_df)

        # Render the result page with the prediction
        return render(request, 'rent_test.html', {'prediction': prediction[0]})
    else:
        # Render the input form page if not a POST request
        return render(request, 'regression.html')
