import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('credit_data.csv')

# Preprocess the data
# For simplicity, we'll just encode the categorical variables
df['SEX'] = df['SEX'].map({'M': 0, 'F': 1})
df['EDUCATION'] = df['EDUCATION'].map({'Unknown': 0, 'High School': 1, 'University': 2, 'Graduate school': 3})
df['MARRIAGE'] = df['MARRIAGE'].map({'Other': 0, 'Single': 1, 'Married': 2})

# Split features and target variable
X = df.drop(columns=['ID', 'default'])
y = df['default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the results
feature_importances = rf_classifier.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
