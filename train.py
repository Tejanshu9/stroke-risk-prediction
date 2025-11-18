#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ======================================================
# PARAMETERS
# ======================================================
output_file = "stroke_model.bin"

BEST_PARAMS = {
    "max_depth": 3,
    "min_samples_split": 2,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": 1
}

# ======================================================
# LOAD DATA
# ======================================================
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Remove ID column
df = df.drop(columns=["id"])

# ======================================================
# FIX MISSING BMI
# ======================================================
bmi_median = df["bmi"].median()
df["bmi"] = df["bmi"].fillna(bmi_median)

# ======================================================
# TRAIN / TEST SPLIT (80/20) — like the notebook
# ======================================================
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

y_full_train = df_full_train["stroke"].values
df_full_train = df_full_train.drop(columns=["stroke"])

y_test = df_test["stroke"].values
df_test = df_test.drop(columns=["stroke"])

# ======================================================
# TRAINING FUNCTION
# ======================================================
def train_decision_tree(df, y, params):

    train_dicts = df.to_dict(orient="records")

    # One-hot encoding
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(train_dicts)

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Decision Tree with best hyperparameters
    dt = DecisionTreeClassifier(**params)
    dt.fit(X, y)

    return dv, scaler, dt

# ======================================================
# TRAIN FINAL MODEL
# ======================================================
print("Training Decision Tree model...")

dv, scaler, model = train_decision_tree(df_full_train, y_full_train, BEST_PARAMS)

print("Model training completed!")

# ======================================================
# SAVE MODEL + DV + SCALER + MEDIAN
# ======================================================
with open(output_file, "wb") as f_out:
    pickle.dump((dv, scaler, model, bmi_median), f_out)

print(f"Model saved to {output_file}")
