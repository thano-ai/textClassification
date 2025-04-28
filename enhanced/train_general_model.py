import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('newDS.csv')

# Preprocessing
df['General Category'] = df['General Category'].fillna('non-technical')
df['Title'] = df['Title'].fillna('')
df['Content'] = df['Content'].fillna('')
df['text'] = df['Title'] + " " + df['Content']

# Encode general category
df['General Category'] = df['General Category'].apply(lambda x: 1 if str(x).lower() == "technical" else 0)

# Features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['General Category']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("General Category Model Results:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Technical", "Technical"], yticklabels=["Non-Technical", "Technical"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('General Category Confusion Matrix')
plt.show()

# Save model and vectorizer
joblib.dump(model, 'general_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
