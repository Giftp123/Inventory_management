import pandas as pd
import numpy as np
import joblib
import sys
import json
import os

# Load the trained model and feature names
try:
    model_path = os.path.join(os.path.dirname(__file__), 'demand_predictor.joblib')
    model = joblib.load(model_path)
    
    features_path = os.path.join(os.path.dirname(__file__), 'model_features.joblib')
    features = joblib.load(features_path)
except Exception as e:
    print(json.dumps({'error': f'Error loading model or features: {e}'}))
    sys.exit(1)

def predict_demand(item_id, date_str, model, features):
    try:
        # Create a DataFrame for the input
        input_data = pd.DataFrame(columns=features)
        
        # Initialize with zeros for one-hot encoded features
        input_data.loc[0] = 0
        
        # Process the date string
        date = pd.to_datetime(date_str)
        input_data['day_of_year'] = date.dayofyear
        input_data['day_of_week'] = date.dayofweek
        input_data['month'] = date.month
        input_data['year'] = date.year
        
        # Set the item_id one-hot encoding (convert to uppercase)
        item_col = f'item_{item_id.upper()}'
        if item_col in input_data.columns:
            input_data[item_col] = 1
        else:
            # Handle cases where the item_id was not in the training data
            # For simplicity, we'll just predict 0, or you could use a default/average item
            print(json.dumps({'error': f'Item ID {item_id} not recognized by the model.'}))
            return None

        # Make prediction
        prediction = model.predict(input_data[features])[0]
        return max(0, int(prediction)) # Demand cannot be negative
    except Exception as e:
        print(json.dumps({'error': f'Error during prediction: {e}'}))
        return None

if __name__ == '__main__':
    # Expecting item_id and date as command line arguments
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Usage: python predict.py <item_id> <date_string>'}))
        sys.exit(1)
    
    item_id = sys.argv[1]
    date_str = sys.argv[2]
    
    predicted_demand = predict_demand(item_id, date_str, model, features)
    
    if predicted_demand is not None:
        print(json.dumps({'item_id': item_id, 'date': date_str, 'predicted_demand': predicted_demand}))
