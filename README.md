# 🧠 Stroke Risk Prediction

A machine-learning project that predicts the probability of a person suffering a stroke based on medical and demographic features.

The project includes:

- Full EDA + model training in Jupyter Notebook  
- Logistic Regression, Decision Tree, and Random Forest evaluation  
- Final model selection (Decision Tree)  
- Model serialization (`stroke_model.bin`)  
- REST API service using Flask  
- Production-grade serving using Gunicorn + Docker  
- Fly.io deployment  

---

## 📌 1. Problem Description  

The goal of this project is to **predict the likelihood of a patient having a stroke** using various medical and demographic factors such as:

- Age  
- Hypertension  
- Heart Disease  
- Smoking Status  
- BMI  
- Average Glucose Level  
- Work Type  
- Residence Type  
- Marital Status  

This is an **imbalanced classification problem** (≈5% stroke cases).  
Therefore, evaluation focuses on:

- ROC-AUC  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

For medical applications, **high recall is preferred** (catch most stroke-risk cases), and lower precision is acceptable.

The final deployed model is a **Decision Tree Classifier** trained with recall-focused evaluation.

---

## 📌 2. Dataset  

Dataset used: **Healthcare Stroke Prediction Dataset**  
File: `healthcare-dataset-stroke-data.csv`

Contains **5110 rows** and **12 columns**.  
The target variable is:  
`stroke` → (0 = no stroke, 1 = stroke)

---

## 📌 3. Project Structure

```bash
stroke-risk-prediction/
│
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── train.py                    # trains the final Decision Tree model
├── predict.py                  # Flask API service for inference
├── stroke_model.bin            # serialized model (dv + scaler + DT model + BMI median)
├── healthcare-dataset-stroke-data.csv
└── notebook.ipynb              # EDA + model development + evaluation
```

---

## 📌 4. Model Training Pipeline

### ✔ Steps Performed

---

### **1. Data Cleaning**
- Missing BMI values imputed using **median**  
- `stroke` column extracted as the target variable  

---

### **2. Train / Validation / Test Split**
- Dataset split into **80% Train** and **20% Test**  
- Test set kept **untouched** for final evaluation  
- Validation used earlier during **Logistic Regression threshold tuning**  

---

### **3. K-Fold Cross-Validation (Model Selection)**
Applied **5-Fold Cross Validation** to compare the following models:

- **Logistic Regression**  
- **Decision Tree**  
- **Random Forest**  

**Metrics evaluated during K-Fold:**
- **ROC-AUC**  
- **Precision**  
- **Recall**  
- **F1-score**  

4. **Feature Encoding**  
   - One-hot encoding for categorical columns  
   - Numerical columns scaled using `StandardScaler`  

5. **Decision Tree Model**  
   - The final deployed model is a **Decision Tree Classifier** trained with the best hyperparameters selected from K-Fold cross-      validation and recall-focused evaluation.


6. **Final Model Training**  
   - Model trained on the full training dataset  
   - Exported as `stroke_model.bin` using `pickle`

### Saved Model Contains  
- `dv` → DictVectorizer  
- `scaler` → StandardScaler  
- `model` → Logistic Regression  
- `bmi_median` → For imputing missing BMI  

---
## 📌 Model Selection Summary

For this project, three different machine learning models were trained and evaluated:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

Since this is a **medical screening problem**, the priority is:

- **High Recall** — important to correctly identify as many stroke-risk patients as possible  
- **Lower Precision is acceptable** — false positives are not harmful compared to missing a real stroke case

During evaluation:

- **The Decision Tree achieved the highest recall**, making it the most suitable model for early-risk medical detection.
- **ROC-AUC scores were similar for all three models**, showing that each model captured the underlying pattern equally well.
- Detailed metrics (ROC-AUC, Precision, Recall, F1-Score, Confusion Matrices) for each model are included in **notebooks.ipynb**.

Based on recall performance and overall interpretability, the **Decision Tree model** was chosen as the final model for deployment.


## 📌 5. Running the Project Locally  

### **1️⃣ Install dependencies**
```bash
pipenv install
pipenv shell
```

2️⃣ Train the model
```bash
python train.py
```

3️⃣ Start API locally
```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```
4️⃣ Test with curl

```bash
curl -X POST http://0.0.0.0:9696/predict \
    -H "Content-Type: application/json" \
    -d '{
        "gender": "male",
        "age": 67,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "yes",
        "work_type": "private",
        "residence_type": "urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked"
    }'
```

## 📌 6. Running with Docker

1️⃣ Build the Docker image
```bash
docker build -t stroke-service .
```

2️⃣ Run the container
```bash
docker run -p 9696:9696 stroke-service
```

3️⃣ Test API
```bash
curl -X POST http://0.0.0.0:9696/predict \
    -H "Content-Type: application/json" \
    -d '{
        "gender": "male",
        "age": 67,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "yes",
        "work_type": "private",
        "residence_type": "urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked"
    }'
```
## 📌 7. Deployment (Fly.io)

### 🔗 **Live API Endpoint**
[https://stroke-risk-prediction.fly.dev/predict](https://stroke-risk-prediction.fly.dev/predict)

### 🌐 Example Request (Live)
```bash
curl -X POST https://stroke-risk-prediction.fly.dev/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```
### 📸 Screenshot of successful cloud prediction
<img width="954" height="244" alt="Screenshot 2025-11-18 at 6 25 12 AM" src="https://github.com/user-attachments/assets/ba80ec39-8378-4a30-9d79-53891c5be46d" />

