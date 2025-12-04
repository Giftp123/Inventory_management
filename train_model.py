import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# 1. Generate Synthetic Data
np.random.seed(42)

# Define a date range
dates = pd.date_range(start='2022-01-01', periods=365*2, freq='D') # 2 years of daily data

# Define a few item IDs
item_ids = [f'ITEM_{i:03d}' for i in range(1, 11)] # 10 items

data = []
for item_id in item_ids:
    # Simulate a base demand, seasonality, and randomness
    base_demand = np.random.randint(50, 200)
    
    for i, date in enumerate(dates):
        seasonality = 1 + 0.3 * np.sin(i / 365 * 2 * np.pi + np.pi/2) # Annual cycle
        trend = 1 + (i / (365*2)) * 0.5 # gradually increasing demand
        
        # Random noise
        noise = np.random.normal(0, 10)
        
        # Demand calculation
        demand = max(0, int(base_demand * seasonality * trend + noise))
        
        data.append({'date': date, 'item_id': item_id, 'demand': demand})

df = pd.DataFrame(data)

# Save raw data for inspection 
df.to_csv('inventory_demand.csv', index=False)
print("Synthetic data generated and saved to inventory_demand.csv")


# Summary
print("\n--- Data Summary ---")
print(df['demand'].describe())
print("--------------------\n")

# Visualization ---
import matplotlib.pyplot as plt

print("Generating demand visualization...")
plt.figure(figsize=(15, 8))

# Plot demand for the first 3 items to show trends and seasonality
for item_id in item_ids[:3]:
    subset = df[df['item_id'] == item_id]
    plt.plot(subset['date'], subset['demand'], label=item_id)

plt.title('Demand Over Time for Sample Items')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
visualization_path = 'demand_visualization.png'
plt.savefig(visualization_path)
print(f"Visualization saved to {visualization_path}")


# 2. Preprocessing for Model Training
# Convert date to numerical features (e.g., day of week, day of year, month, year)
df['day_of_year'] = df['date'].dt.dayofyear
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Convert item_id to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['item_id'], prefix='item')

# Define features (X) and target (y)
features = ['day_of_year', 'day_of_week', 'month', 'year'] + [col for col in df.columns if 'item_' in col]
target = 'demand'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Selection and Training
model = LinearRegression()
model.fit(X_train, y_train)



# 4. Save the trained model and feature names
model_dir = 'inventory_management_system/ml-model'
os.makedirs(model_dir, exist_ok=True) # Ensure the directory exists

model_path = os.path.join(model_dir, 'demand_predictor.joblib')
joblib.dump(model, model_path)
print(f"Trained model saved to {model_path}")

features_path = os.path.join(model_dir, 'model_features.joblib')
joblib.dump(features, features_path)
print(f"Model features saved to {features_path}")

print("ML model training and saving complete.")