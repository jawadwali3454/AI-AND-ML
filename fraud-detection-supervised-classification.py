# Generated from: fraud-detection-supervised-classification.ipynb
# Converted at: 2026-05-04T08:18:13.872Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

# Load dataset (place creditcard.csv in same folder)
df = pd.read_csv('/kaggle/input/datasets/jawadwali/fraud-detection/creditcard (1).csv')
df.head()

df.shape, df['Class'].value_counts()

# Class imbalance
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()

fraud_pct = df['Class'].mean()*100
print(f'Fraud Percentage: {fraud_pct:.3f}%')

# Feature Engineering
df['Amount_Scaled'] = StandardScaler().fit_transform(df[['Amount']])
df = df.drop(columns=['Amount'])  # Time can be kept or dropped
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape, X_test.shape)

# Apply SMOTE on training data
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print('Before:', y_train.value_counts().to_dict())
print('After :', pd.Series(y_train_sm).value_counts().to_dict())

def evaluate_model(model, Xtr, ytr, Xte, yte, name):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:,1] if hasattr(model,'predict_proba') else pred

    print('='*60)
    print(name)
    print(confusion_matrix(yte, pred))
    print(classification_report(yte, pred, digits=4))
    print('ROC-AUC :', round(roc_auc_score(yte, proba),4))
    print('PR-AUC  :', round(average_precision_score(yte, proba),4))

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
evaluate_model(lr, X_train_sm, y_train_sm, X_test, y_test, 'Logistic Regression')

# KNN (sampled training for speed if needed)
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn, X_train_sm, y_train_sm, X_test, y_test, 'KNN')



# Naive Bayes
nb = GaussianNB()
evaluate_model(nb, X_train_sm, y_train_sm, X_test, y_test, 'Gaussian Naive Bayes')