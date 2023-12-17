from django.shortcuts import render
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from .milk_test import load_model
import mysql.connector
import seaborn as sns
import matplotlib.pyplot as plt
import urllib, base64
import io
from .models import rent_prediction,milk_prediction

def index(request):
    # Database connection
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="prediction"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM milknew") 
    rows = cursor.fetchall()
    columns = [i[0] for i in cursor.description]
    data = pd.DataFrame(rows, columns=columns)

    # Visualization Code
    sns.set(style="whitegrid")

    # Plot for Class Distribution in Grade
    plt.figure(figsize=(7, 6))  # Adjust size as needed
    sns.countplot(x='Grade', data=data)
    plt.title('Class Distribution in Grade')

    # Convert first plot to HTML
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    string1 = base64.b64encode(buf1.read())
    uri1 = urllib.parse.quote(string1)
    plt.close()

    # Plot for Correlation Among Features
    plt.figure(figsize=(7, 6))  # Adjust size as needed
    numerical_data = data.drop(['Grade'], axis=1)  # Adjust based on your data
    corr = numerical_data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Among Features')

    # Convert second plot to HTML
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    string2 = base64.b64encode(buf2.read())
    uri2 = urllib.parse.quote(string2)
    plt.close()

    #HOUSE RENT
    # Fetch house rent data from the database
    cursor.execute("SELECT * FROM house_rent_dataset") 
    rows = cursor.fetchall()
    columns = [i[0] for i in cursor.description]
    house_rent_data = pd.DataFrame(rows, columns=columns)

    # Visualization Code
    sns.set(style="whitegrid")
    
    # Plot 1: Rent Distribution
    plt.figure(figsize=(7, 6))
    sns.histplot(house_rent_data['Rent'], kde=True)
    plt.title('Rent Distribution')

    #Convert to HTML
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', bbox_inches='tight')
    buf1.seek(0)
    uri3 = urllib.parse.quote(base64.b64encode(buf1.read()))
    plt.close()
    
    # Plot 2: Rent vs Size
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=house_rent_data, x='Size', y='Rent')
    plt.title('Rent vs Size')

    # Convert second plot to HTML
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    string2 = base64.b64encode(buf2.read())
    uri4 = urllib.parse.quote(string2)
    plt.close()
    #3. Rent by City - Bar Plot
    plt.figure(figsize=(10, 6))
    average_rent_by_city = house_rent_data.groupby('City')['Rent'].mean().sort_values()
    sns.barplot(x=average_rent_by_city.values, y=average_rent_by_city.index)
    plt.title('Average Rent by City')
    #Convert to HTML
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png')
    buf3.seek(0)
    # string2 = base64.b64encode(buf3.read())
    uri5 = urllib.parse.quote(base64.b64encode(buf3.read()))
    plt.close()

    # 4. Furnishing Status vs Rent - Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Furnishing_Status', y='Rent', data=house_rent_data)
    plt.title('Furnishing Status vs Rent')
    
    #Convert to HTML
    buf4 = io.BytesIO()
    plt.savefig(buf4, format='png')
    buf4.seek(0)
    # string2 = base64.b64encode(buf3.read())
    uri6 = urllib.parse.quote(base64.b64encode(buf4.read()))
    plt.close()

    #5. BHK Distribution - Bar Plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x='BHK', data=house_rent_data)
    plt.title('BHK Distribution')

    #Convert to HTML
    buf5 = io.BytesIO()
    plt.savefig(buf5, format='png')
    buf5.seek(0)
    # string2 = base64.b64encode(buf3.read())
    uri7 = urllib.parse.quote(base64.b64encode(buf5.read()))
    plt.close()
    # Pass both URIs to the template
    return render(request, 'index.html', 
    {
        'data_uri_1': uri1, 
        'data_uri_2': uri2, 
        'data_uri_3': uri3, 
        'data_uri_4': uri4, 
        'data_uri_5': uri5,
        'data_uri_6': uri6,
        'data_uri_7': uri7
    })

def regression(request):
    return render(request, 'regression.html')

#REGRESSION
def predict_rent(request):
    if request.method == 'POST':
        # Extract data from POST request
        BHK = int(request.POST.get('BHK'))
        Size = float(request.POST.get('Size'))
        Floor = int(request.POST.get('Floor'))
        Bathroom = int(request.POST.get('Bathroom'))
        FurnishingStatus = request.POST.get('FurnishingStatus')
        AreaType = request.POST.get('AreaType')
        AreaLocality = request.POST.get('AreaLocality')
        City = request.POST.get('City')
        # TenantPreferred = request.POST.get('TenantPreferred')
        # Extract other features similarly

        # Prepare the input data
        input_data = pd.DataFrame({
            'BHK': [BHK],
            'Size': [Size],
            'Floor': [Floor],
            'Bathroom': [Bathroom],
            'Furnishing Status': [FurnishingStatus],
            'Area Type' : [AreaType],
            'Area Locality' : [AreaLocality],
            'City' : [City],
            # 'Tenant Preferred': [TenantPreferred],
            # Add other features
        })

        # Debugging: Print the DataFrame's shape and content
        print("DataFrame shape:", input_data.shape)
        print("DataFrame content:\n", input_data)

        # Load the model
        model = joblib.load('C:\\Users\\welun\\ITD 105\\Regression\\rent_prediction_model.pkl')
        # Make the prediction
        prediction = model.predict(input_data)
        rounded_prediction = round(prediction[0])

        # Save the input data and prediction result to db
        rent_prediction.objects.create(
            BHK=BHK,
            Size=Size,
            Floor=Floor,
            Bathroom=Bathroom,
            FurnishingStatus=FurnishingStatus,
            AreaType=AreaType,
            AreaLocality=AreaLocality,
            City=City,
            PredictedRent=rounded_prediction
        )


        # Render the result
        return render(request, 'rent_test.html', {'prediction': prediction[0]})

    # If not a POST request, render the input form
    return render(request, 'regression.html')


def classification(request):
    return render(request, 'classification.html')

# Load the model and the scaler
model = load_model()
scaler = joblib.load('C:\\Users\\welun\\ITD 105\\Classification\\milk_scaler.pkl')

def predict_milk(request):
    if request.method == 'POST':
        # Retrieve form data with checks
        pH = request.POST.get('pH')
        pH = float(pH) if pH else 0.0  
        Temprature = request.POST.get('Temprature')  #retrieve data
        Temprature = float(Temprature) if Temprature else 0.0  
        Taste = float(request.POST.get('Taste')) if request.POST.get('Taste') else 0.0
        Odor = float(request.POST.get('Odor')) if request.POST.get('Odor') else 0.0
        Fat = float(request.POST.get('Fat')) if request.POST.get('Fat') else 0.0
        Turbidity = float(request.POST.get('Turbidity')) if request.POST.get('Turbidity') else 0.0
        Colour = float(request.POST.get('Colour')) if request.POST.get('Colour') else 0.0
        
        # Convert features to appropriate format
        features = np.array([pH, Temprature, Taste, Odor, Fat, Turbidity, Colour])
        features = features.reshape(1, -1)

        # Scale the features
        features_scaled = scaler.transform(features) #scale, normalize data 

        # Make prediction
        prediction = model.predict(features_scaled)

        # Create a new instance and save the inputs to the database
        milk_prediction.objects.create(
            pH=pH,
            Temprature=Temprature,
            Taste=Taste,
            Odor=Odor,
            Fat=Fat,
            Turbidity=Turbidity,
            Colour=Colour,
            milkquality=prediction[0]
        )
        return render(request, 'milk_test.html', {'prediction': prediction[0]})
    
def show_records(request):
    try:
        # Database connection
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="prediction"
        )
        cursor = conn.cursor(dictionary=True)  # Use dictionary cursor
        cursor.execute("SELECT * FROM milknew")
        milknews = cursor.fetchall()
        cursor.close()
        conn.close()

        # Check if data is empty
        if not milknews:
            return render(request, 'show.html', {'error': 'No data found in the database.'})

        return render(request, 'show.html', {'milknews': milknews})

    except mysql.connector.Error as err:
        return render(request, 'show.html', {'error': f'Database error: {err}'})


def show_rent(request):
    try:
        # Database connection
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="prediction"
        )
        cursor = conn.cursor(dictionary=True)  # Use dictionary cursor
        cursor.execute("SELECT * FROM house_rent_dataset")
        rent = cursor.fetchall()
        cursor.close()
        conn.close()

        # Check if data is empty
        if not rent:
            return render(request, 'showreg.html', {'error': 'No data found in the database.'})

        return render(request, 'showreg.html', {'rent': rent})

    except mysql.connector.Error as err:
        return render(request, 'showreg.html', {'error': f'Database error: {err}'})

