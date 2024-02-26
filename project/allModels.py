import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def common(data_list, model, scaler, features):
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

def is_Graduate_model_full(data_list):
        # Load the model
    with open('models/model_7.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the scaler
    with open('models/scaler_7.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the list of features
    with open('models/features_7.pkl', 'rb') as f:
        features = pickle.load(f)

    print('Model, scaler, and features loaded correcty.')

    
    prediction = common(data_list, model, scaler, features)

    return prediction

def is_Graduate_model_1(data_list):
    # Load the model
    with open('models/model_1.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the scaler
    with open('models/scaler_1.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the list of features
    with open('models/features_1.pkl', 'rb') as f:
        features = pickle.load(f)
    
    print('Model, scaler, and features loaded correcty.')

    prediction = common(data_list, model, scaler, features)

    return prediction

def is_Graduate_model_2(data_list):
    # Load the model
    with open('models/model_2.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the scaler
    with open('models/scaler_2.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the list of features
    with open('models/features_2.pkl', 'rb') as f:
        features = pickle.load(f)

    print('Model, scaler, and features loaded correcty.')
    
    prediction = common(data_list, model, scaler, features)

    return prediction

def is_Graduate_model_3(data_list):
    # Load the model
    with open('models/model_3.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the scaler
    with open('models/scaler_3.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the list of features
    with open('models/features_3.pkl', 'rb') as f:
        features = pickle.load(f)

    print('Model, scaler, and features loaded correcty.')
    
    prediction = common(data_list, model, scaler, features)

    return prediction

def is_Graduate_model_4(data_list):
    # Load the model
    with open('models/model_4.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the scaler
    with open('models/scaler_4.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the list of features
    with open('models/features_4.pkl', 'rb') as f:
        features = pickle.load(f)
    
    print('Model, scaler, and features loaded correcty.')

    prediction = common(data_list, model, scaler, features)

    return prediction

def is_Graduate_model_5(data_list):
    # Load the model
    with open('models/model_5.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the scaler
    with open('models/scaler_5.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the list of features
    with open('models/features_5.pkl', 'rb') as f:
        features = pickle.load(f)
    
    print('Model, scaler, and features loaded correcty.')

    prediction = common(data_list, model, scaler, features)

    return prediction

def is_Graduate_model_6(data_list):
    # Load the model
    with open('models/model_6.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load the scaler
    with open('models/scaler_6.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the list of features
    with open('models/features_6.pkl', 'rb') as f:
        features = pickle.load(f)
    
    print('Model, scaler, and features loaded correcty.')

    prediction = common(data_list, model, scaler, features)

    return prediction







