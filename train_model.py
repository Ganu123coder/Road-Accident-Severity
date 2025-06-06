import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("/Users/harshuu/Desktop/ML_G/dft-road-casualty-statistics-casualty-2023.csv")

# Drop unnecessary or problematic columns
df = df.drop(columns=["accident_index", "accident_reference", "lsoa_of_casualty"])

# Replace invalid values (e.g., -1) with NaN and drop them
df.replace(-1, pd.NA, inplace=True)
df.dropna(inplace=True)

# Define features and target
X = df.drop(columns=["casualty_severity"])
y = df["casualty_severity"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("accident_severity_model.pkl", "wb") as f:
    pickle.dump(model, f)
