import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/spam.csv"
data = pd.read_csv(url, encoding="latin-1").iloc[:, [0, 1]]  # Select only the first two columns
data.columns = ["label", "message"]

# Convert labels (ham → 0, spam → 1)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Function to clean text data
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Apply text cleaning
data["message"] = data["message"].apply(preprocess_text)

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(data["message"], data["label"], test_size=0.2, random_state=42)

# Convert text data into numerical format using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Function to predict spam or ham
def predict_message(msg):
    msg = preprocess_text(msg)  # Clean the message
    msg_vec = vectorizer.transform([msg])  # Convert to vector format
    prediction = model.predict(msg_vec)
    return "Spam" if prediction[0] == 1 else "Ham"

# Test with user input
while True:
    user_input = input("\nEnter a message to check (or type 'exit' to stop): ")
    if user_input.lower() == "exit":
        break
    print("Prediction:", predict_message(user_input))
