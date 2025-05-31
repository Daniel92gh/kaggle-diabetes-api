import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# project path 
path = Path(__file__).parent.parent

# load dataset
data_path = path / 'data' / 'kaggle_diabetes.csv'
df = pd.read_csv(data_path)
df.rename(columns={'DiabetesPedigreeFunction': 'DPF'}, inplace=True)
print("Data is loaded !")

# replace invalid zeros with NaN
replace_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[replace_cols] = df[replace_cols].replace(0, np.nan)

# fill NaN values with proper statistics
df['Glucose'].fillna(df['Glucose'].mean(), inplace=True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace=True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace=True)
df['Insulin'].fillna(df['Insulin'].median(), inplace=True)
df['BMI'].fillna(df['BMI'].median(), inplace=True)

# x and y dfs
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# train model
classifier = RandomForestClassifier(n_estimators=100, random_state=123)
classifier.fit(X_train_scaled, y_train)
print("Model training finished !")

# model evaluation
train_pred = classifier.predict(X_train_scaled)
test_pred = classifier.predict(X_test_scaled)

train_acc = accuracy_score(y_train, train_pred) * 100
test_acc = accuracy_score(y_test, test_pred) * 100

print(f"Training Accuracy: {train_acc:.3f}%")
print(f"Test Accuracy: {test_acc:.3f}%")

# save model
model_path = path / 'model' / 'rf_classifier.pkl'
with open(model_path, 'wb') as model_file:
    pickle.dump(classifier, model_file)

print(f"Model is saved at {model_path}")    

# save scaler
scaler_path = path / 'scaler' / 'scaler.pkl'
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"Scaler is saved at {scaler_path}")
