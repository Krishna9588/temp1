import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Manually create a dataset
data = {
    'packet_rate': [100, 150, 2000, 3000, 250, 4000, 4500, 90, 6000, 7000, 300, 50],
    'unique_ips': [5, 10, 500, 700, 15, 1000, 1200, 3, 1500, 1800, 20, 2],
    'bytes_per_request': [500, 600, 50, 45, 700, 40, 30, 800, 25, 20, 750, 900],
    'label': [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0]  # 0 = Normal, 1 = DDoS
}

# Convert to DataFrame
df = pd.DataFrame(data)

# 2. Split into features and labels
X = df.drop(columns=['label'])  # Features
y = df['label']  # Target

# 3. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predict on test data
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("Model trained successfully!\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Dynamic User Input for Traffic Classification
while True:
    print("\nEnter network traffic details to classify (or type 'exit' to quit):")
    try:
        packet_rate = float(input("Enter Packet Rate (packets/sec): "))
        unique_ips = float(input("Enter Unique IP Count: "))
        bytes_per_request = float(input("Enter Bytes Per Request: "))

        # Predict traffic type
        input_data = np.array([[packet_rate, unique_ips, bytes_per_request]])
        prediction = model.predict(input_data)[0]
        
        # Display result
        result = "🚨 DDoS Attack Detected!" if prediction == 1 else "✅ Normal Traffic"
        print(f"\nClassification Result: {result}\n")

    except ValueError:
        print("Invalid input! Please enter numerical values.")
    
    # Exit condition
    if input("Do you want to test another entry? (yes/no): ").strip().lower() != 'yes':
        print("Exiting program. Goodbye!")
        break
