import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("dataset/documents.csv")

# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])

# Labels
y = data["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "model/risk_classifier.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model trained successfully!")