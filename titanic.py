# Generated from: titanic.ipynb
# Converted at: 2026-05-07T10:46:53.290Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# <a href="https://colab.research.google.com/github/jawadwali3454/AI-AND-ML/blob/main/titanic.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


from google.colab import files

uploaded = files.upload()

import os

os.listdir()
!unzip "archive (6).zip"

import pandas as pd

# Load the dataset
df = pd.read_csv('SVMtrain.csv')

# Display the first 5 rows of the DataFrame
print("First 5 rows of the dataset:")
display(df.head())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])

df.head()

# checking encode
df['Sex'].unique()



# Display descriptive statistics
print("\nDescriptive Statistics:")
display(df.describe(include='all'))

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Based on the output of df.isnull().sum(), we can handle missing values.
# For example, let's impute missing 'Age' values with the median.
# (Assuming 'Age' is the primary column with missing values based on typical Titanic datasets)
# If other columns have missing values, a similar approach or other imputation strategies can be applied.

# Check if 'Age' has missing values and impute if necessary
if df['Age'].isnull().any():
    median_age = df['Age'].median()
    df['Age'].fillna(median_age, inplace=True)
    print(f"\nMissing 'Age' values imputed with median: {median_age}")
else:
    print("\nNo missing 'Age' values found.")

# Re-check for missing values after imputation
print("\nMissing values after imputation:")
print(df.isnull().sum())

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
# X = df.drop('Survived', axis=1) # Already done in cell wFEbu3AFSGHW
# y = df['Survived']             # Already done in cell wFEbu3AFSGHW

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# ### Applying SMOTE for Data Balancing
# 
# Since the dataset was unbalanced, we will apply the Synthetic Minority Over-sampling Technique (SMOTE) to the training data to balance the classes. This helps prevent the models from being biased towards the majority class.


from imblearn.over_sampling import SMOTE

smt = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)

print(f"Shape of X_train before SMOTE: {X_train.shape}")
print(f"Shape of y_train before SMOTE: {y_train.shape}")
print(f"Class distribution in y_train before SMOTE:\n{y_train.value_counts()}")

print(f"\nShape of X_train after SMOTE: {X_train_resampled.shape}")
print(f"Shape of y_train after SMOTE: {y_train_resampled.shape}")
print(f"Class distribution in y_train after SMOTE:\n{y_train_resampled.value_counts()}")

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize the SVM classifier
svm_model_smote = SVC(random_state=42)

# Train the model on SMOTE-resampled data
svm_model_smote.fit(X_train_resampled, y_train_resampled)

# Make predictions on the original test set
y_pred_svm_smote = svm_model_smote.predict(X_test)

# Evaluate the model
print("\n--- Support Vector Machine (SVM) Model Evaluation with SMOTE ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm_smote):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm_smote))

# Confusion Matrix
cm_svm_smote = confusion_matrix(y_test, y_pred_svm_smote)

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm_svm_smote, annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.tree import DecisionTreeClassifier

# Initialize the Decision Tree classifier
dt_model_smote = DecisionTreeClassifier(random_state=42)

# Train the model on SMOTE-resampled data
dt_model_smote.fit(X_train_resampled, y_train_resampled)

# Make predictions on the original test set
y_pred_dt_smote = dt_model_smote.predict(X_test)

# Evaluate the model
print("\n--- Decision Tree Classifier Model Evaluation (with SMOTE) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt_smote):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt_smote))

# Confusion Matrix
cm_dt_smote = confusion_matrix(y_test, y_pred_dt_smote)

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm_dt_smote, annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree Confusion Matrix ")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
rf_model_smote = RandomForestClassifier(random_state=42)

# Train the model on SMOTE-resampled data
rf_model_smote.fit(X_train_resampled, y_train_resampled)

# Make predictions on the original test set
y_pred_rf_smote = rf_model_smote.predict(X_test)

# Evaluate the model
print("\n--- Random Forest Classifier Model Evaluation ")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_smote):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf_smote))

# Confusion Matrix
cm_rf_smote = confusion_matrix(y_test, y_pred_rf_smote)

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm_rf_smote, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix ")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Get metrics for each SMOTE-trained model
svm_metrics_smote = get_metrics(y_test, y_pred_svm_smote, 'SVM ')
dt_metrics_smote = get_metrics(y_test, y_pred_dt_smote, 'Decision Tree')
rf_metrics_smote = get_metrics(y_test, y_pred_rf_smote, 'Random Forest')

# Create a DataFrame for comparison
comparison_df_smote = pd.DataFrame([svm_metrics_smote, dt_metrics_smote, rf_metrics_smote])

# Set Model name as index for better readability
comparison_df_smote = comparison_df_smote.set_index('Model')

print("\n--- Model Performance Comparison (with SMOTE) ---")
display(comparison_df_smote)

import numpy as np

# Get accuracies from the SMOTE-trained models
accuracy_svm_smote = accuracy_score(y_test, y_pred_svm_smote)
accuracy_dt_smote = accuracy_score(y_test, y_pred_dt_smote)
accuracy_rf_smote = accuracy_score(y_test, y_pred_rf_smote)

models_smote = ['SVM ', 'Decision Tree ', 'Random Forest']
accuracies_smote = [accuracy_svm_smote, accuracy_dt_smote, accuracy_rf_smote]

plt.figure(figsize=(10, 6))
sns.barplot(x=models_smote, y=accuracies_smote, palette='viridis', hue=models_smote, legend=False)
plt.title('Comparison of Model Accuracies ')
plt.ylabel('Accuracy')
plt.ylim(0, 1) # Accuracy is between 0 and 1
for index, value in enumerate(accuracies_smote):
    plt.text(index, value + 0.02, f'{value:.4f}', ha='center')
plt.show()

import os

os.listdir()