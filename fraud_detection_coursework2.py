
# CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('creditcard.csv')

print(df.head())
print("Dataset Shape:", df.shape)

# Data Understanding
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Fraud Distribution Graph
plt.figure(figsize=(6,5))
sns.countplot(x='Class', data=df)
plt.title('Distribution of Fraud and Normal Transactions')
plt.savefig('fraud_distribution.png')
plt.close()

# Transaction Amount Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Transaction Amount Distribution')
plt.savefig('transaction_amount_distribution.png')
plt.close()

# Correlation Heatmap
plt.figure(figsize=(15,12))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

# Data Preprocessing
scaler = StandardScaler()

df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Amount', 'Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# Handle Class Imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred_log))

# Decision Tree
DT = DecisionTreeClassifier(random_state=42)
DT.fit(X_train, y_train)

y_pred_dt = DT.predict(X_test)
y_prob_dt = DT.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred_dt))

# Random Forest
RF = RandomForestClassifier(n_estimators=100, random_state=42)
RF.fit(X_train, y_train)

y_pred_rf = RF.predict(X_test)
y_prob_rf = RF.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred_rf))

# Isolation Forest
iso = IsolationForest(contamination=0.0017, random_state=42)
iso.fit(X_train)

y_pred_iso = iso.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)

print(classification_report(y_test, y_pred_iso))

# ROC Curve
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8,6))
plt.plot(fpr_log, tpr_log, label='Logistic Regression')
plt.plot(fpr_dt, tpr_dt, label='Decision Tree')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

# Feature Importance
importance = RF.feature_importances_
features = X.columns

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(10,6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance.head(10)
)

plt.title('Top 10 Important Features')
plt.savefig('feature_importance.png')
plt.close()

# Model Comparison
results = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Decision Tree',
        'Random Forest',
        'Isolation Forest'
    ],
    'Accuracy': [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_dt),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_iso)
    ]
})

results.to_csv('model_results.csv', index=False)

print('Project Completed Successfully!')
