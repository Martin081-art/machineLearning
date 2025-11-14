# train_model_binary.py

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# --- Load your cleaned dataset ---
df = pd.read_csv("cleaned_student_data.csv")

# --- Merge Enrolled (1) and Graduate (2) into Success (1) ---
df['Target_binary'] = df['Target_num'].apply(lambda x: 0 if x == 0 else 1)

# --- Features and target ---
X = df.drop(['Target_num', 'Target_binary'], axis=1)
y = df['Target_binary']

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Initialize CatBoost classifier ---
model = CatBoostClassifier(
    iterations=400,
    depth=6,
    learning_rate=0.1,
    verbose=0,
    random_state=42
)

# --- Train the model ---
model.fit(X_train, y_train)

# --- Evaluate on test set ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Binary Classification Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Save the trained model ---
with open("best_catboost_model_binary.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n💾 Binary CatBoost model saved as 'best_catboost_model_binary.pkl'")
