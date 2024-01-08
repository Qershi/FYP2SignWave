from django.shortcuts import render
import firebase_admin
from firebase_admin import credentials, db
from django.http import JsonResponse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from scipy import stats
import joblib
#from django.shortcuts import render

def home_view(request):
    return render(request, 'main.html')

def tutorial_view(request):
    return render(request, 'tutorial.html')

def design_view(request):
    return render(request, 'design.html')

# Function to initialize Firebase and retrieve data
def initialize_firebase_and_get_data():
    # Check if Firebase app is already initialized
    if not firebase_admin._apps:
        # Initialize Firebase with service account credentials
        cred = credentials.Certificate('signlanguage/credentials/esp32glovedataset-firebase-adminsdk-fevrw-f03886b00b.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://esp32glovedataset-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })
    return db.reference('test')  # Replace with your database path

# Load your KNN classifier and any other necessary preprocessing steps
knn_classifier = joblib.load('signlanguage/knn_model/model.pkl')



def predict_view(request):
    # Initialize Firebase and retrieve data
    ref = initialize_firebase_and_get_data()

    # Retrieve data from Firebase
    json_data = ref.get()

    # Check if data was successfully retrieved
    if json_data:

            # Remove the "Word" key from the JSON data
        if 'Label' in json_data:
            del json_data['Label']

        # Load the pre-trained scaler
        scaler = joblib.load('signlanguage/knn_model/trained_scaler.pkl')

        # Assuming your JSON data has features that match the original training data
        # Adjust column names as needed
        X = json_data  # Use JSON data directly as features

        # Convert the JSON data to a NumPy array
        X_array = np.array(list(X.values())).reshape(1, -1)

        # Scale the features
        X_scaled = scaler.transform(X_array)

        # Make predictions using the trained KNN classifier
        prediction = knn_classifier.predict(X_scaled)


        # Pass the prediction value to the HTML template
        context = {'prediction': prediction[0]}

        # Render the HTML template with the prediction value
        return render(request, 'prediction.html', context)
    else:
        return JsonResponse({'error': 'Failed to retrieve data from Firebase.'}, status=400) 