from sklearn.model_selection import train_test_split
import numpy as np

#  data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # Feature data
y = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # Target labels

#  training (80%) and test (20%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training features:\n", X_train)
print("Test features:\n", X_test)
print("Training labels:\n", y_train)
print("Test labels:\n", y_test)
