import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv('house_data.csv')
X = data.drop('Price', axis=1)  # All features except the target variable (Price)
y = data['Price']  # Target variable (Price)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r_squared)
# Replace the values below with the actual values of the new data
new_data = pd.DataFrame({
    'Area': [1200, 1500, 1800],
    'Bedrooms': [2, 3, 4],
    # Add other features from your dataset if applicable
})

predicted_prices = model.predict(new_data)
print("Predicted Prices:", predicted_prices)
