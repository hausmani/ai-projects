# Step 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create the Dataset
data = {
    'study_hours': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'attendance': [60, 70, 65, 80, 90, 85, 95, 96, 98, 100],
    'past_score': [50, 55, 60, 65, 70, 75, 78, 80, 85, 88],
    'final_score': [55, 60, 62, 70, 76, 78, 84, 88, 90, 95]
}

df = pd.DataFrame(data)

# LinearRegression()	A regression model that finds the best-fit straight line through the data (simple and interpretable).
# train_test_split()	Splits the dataset into training and testing sets to evaluate model performance.
# mean_squared_error()	Measures how close the predicted scores are to the actual scores.
# r2_score()	Represents how well the model explains the variability in the data (1 = perfect, 0 = bad).

# Why?
# Because predicting a number = regression → LinearRegression is ideal for a beginner-friendly numeric prediction task.

# Step 3: Visualize the Data (optional)
plt.scatter(df['study_hours'], df['final_score'])
plt.xlabel("Study Hours")
plt.ylabel("Final Exam Score")
plt.title("Study Hours vs Final Score")
plt.grid(True)
plt.show()

# Step 4: Prepare Features and Target
X = df[['study_hours', 'attendance', 'past_score']]  # Features
y = df['final_score']  # Target

# Step 5: Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make Predictions on Test Set
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 9: Predict a New Student's Final Score
new_student = [[8, 90, 75]]  # 8 study hours, 90% attendance, 75 past score
predicted_score = model.predict(new_student)
print("\nPrediction for New Student:")
print(f"Predicted Final Exam Score: {predicted_score[0]:.2f}")
