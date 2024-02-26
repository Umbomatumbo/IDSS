import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the model
with open('model_7.pkl', 'rb') as f:
    model = pickle.load(f) 

# Load the scaler
with open('scaler_7.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the list of features
with open('features_7.pkl', 'rb') as f:
    features = pickle.load(f)

print('Model, scaler, and features loaded correcty.')

def is_Graduate_model_full(data_list):
    #check if the data_list is a list
    if not isinstance(data_list, list):
        raise ValueError('data_list must be a list')
    #check if the data_list has the correct number of elements
    if len(data_list[0]) != len(features):
        raise ValueError('data_list must have ',len(features),' elements, bue has ',len(data_list[0]),' elements')
    
    # Create a dataframe with the data list
    df = pd.DataFrame(data_list, columns=features)    
    # Transform the data
    data_scaled = scaler.transform(df)
    # Get the prediction
    prediction = model.predict(data_scaled)
    if(prediction == 1):
        print("The student is prediceted to graduate")
    else:
        print("The student is prediceted to drop out")
    # Return the prediction
    return prediction






