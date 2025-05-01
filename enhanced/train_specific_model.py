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
df = pd.read_csv('NewDS.csv')

# Preprocessing
df['General Category'] = df['General Category'].fillna('non-technical')
# most_frequent = df['Specific Category'].mode()[0]
# df['Specific Category'] = df['Specific Category'].fillna(most_frequent)

# Fill non-technical rows with '_'
df['Specific Category'] = np.where(df['General Category'].str.lower() == 'non-technical',
                                   df['Specific Category'].fillna('_'),
                                   df['Specific Category'])

# Fill technical rows with most frequent technical category (excluding '_')
most_freq_tech = df.loc[(df['General Category'].str.lower() == 'technical') &
                        (df['Specific Category'] != '_'), 'Specific Category'].mode()[0]

df['Specific Category'] = np.where((df['General Category'].str.lower() == 'technical') &
                                   (df['Specific Category'].isna()),
                                   most_freq_tech,
                                   df['Specific Category'])

df['Title'] = df['Title'].fillna('')
df['Content'] = df['Content'].fillna('')
df['text'] = df['Title'] + " " + df['Content']

# Encode general category
df['General Category'] = df['General Category'].apply(lambda x: 1 if str(x).lower() == "technical" else 0)

# Filter only technical
# tech_df = df[df['General Category'] == 1]
tech_df = df[(df['General Category'] == 1) & (df['Specific Category'] != '_')]

if tech_df.shape[0] < 10:
    print(f"Not enough technical samples ({tech_df.shape[0]} found). Specific model training skipped.")
    exit()

# Features
vectorizer = joblib.load('vectorizer.pkl')  # Use the same vectorizer!
X = vectorizer.transform(tech_df['text'])
y = tech_df['Specific Category']

# Encode specific categories
y = y.astype('category')
category_mapping = dict(enumerate(y.cat.categories))
y = y.cat.codes

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Specific Category Model Results:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=category_mapping.values(), yticklabels=category_mapping.values())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Specific Category Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# Save model and category mapping
joblib.dump(model, 'specific_model.pkl')
joblib.dump(category_mapping, 'category_mapping.pkl')
