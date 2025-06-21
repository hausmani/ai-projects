import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'area': [1000, 1500, 1800, 2400, 3000],
    'bedrooms': [2, 3, 3, 4, 5],
    'bathrooms': [1, 2, 2, 3, 3],
    'price': [200000, 300000, 350000, 450000, 600000]
}

df = pd.DataFrame(data)

# LinearRegression()	Fits a straight line to multiple numeric features to predict the price.
# mean_squared_error()	Evaluates how far predictions are from real prices.

# Why?
# just like student predictor
# It also involves numeric predictions,
# So, predicting a number = regression → LinearRegression is ideal for a beginner-friendly numeric prediction task

# Features and target
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction & evaluation
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

# Predict new house price
new_house = [[2200, 3, 2]]
predicted_price = model.predict(new_house)
print("Predicted Price:", predicted_price[0])
