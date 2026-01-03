import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# NOTE:
# Replace 'data.csv' with your LARGE dataset file name
# Required columns: 'text' and 'label' (1 = Real, 0 = Fake)

df = pd.read_csv("data.csv")

X = df["text"]
y = df["label"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
print("Training Accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save model & vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ model.pkl and vectorizer.pkl saved successfully")
