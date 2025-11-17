#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ======================================================
# PARAMETERS
# ======================================================
C = 10   # Best hyperparameter from notebook
output_file = "stroke_model.bin"


# ======================================================
# LOAD RAW DATA
# ======================================================
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Drop ID column (not useful)
df = df.drop(columns=["id"])

# Fix smoking_status "Unknown" → keep as category (model handles it)

# ======================================================
# FIX MISSING VALUES (BMI)
# ======================================================
bmi_median = df["bmi"].median()
df["bmi"] = df["bmi"].fillna(bmi_median)


# ======================================================
# TRAIN / TEST SPLIT (same as notebook)
# 80% train, 20% test
# ======================================================
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# ======================================================
# SEPARATE TARGET
# ======================================================
y_full_train = df_full_train["stroke"].values
df_full_train = df_full_train.drop(columns=["stroke"])

y_test = df_test["stroke"].values
df_test = df_test.drop(columns=["stroke"])


# ======================================================
# FEATURE LISTS (same as notebook)
# ======================================================
numerical = ['age', 'hypertension', 'heart_disease',
             'avg_glucose_level', 'bmi']

categorical = ['gender', 'ever_married', 'work_type',
               'residence_type', 'smoking_status']


# ======================================================
# TRAINING FUNCTION
# ======================================================
def train(df, y, C):

    train_dicts = df.to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(train_dicts)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X, y)

    return dv, scaler, model


# ======================================================
# TRAIN FINAL MODEL
# ======================================================
print("Training final model...")

dv, scaler, model = train(df_full_train, y_full_train, C=C)

print("Model training completed!")


# ======================================================
# SAVE MODEL + DV + SCALER + MEDIAN
# ======================================================
with open(output_file, "wb") as f_out:
    pickle.dump((dv, scaler, model, bmi_median), f_out)

print(f"Model saved to {output_file}")
