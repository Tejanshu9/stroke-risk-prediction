# 🧠 Stroke Risk Prediction

A machine-learning project that predicts the probability of a person suffering a stroke based on medical and demographic features.  
The project includes:

- Full EDA + model training in Jupyter Notebook  
- Logistic Regression model with threshold tuning  
- Model serialization (`stroke_model.bin`)  
- REST API service using Flask  
- Production-grade serving using Gunicorn + Docker  
- Fly.io deployment ready  

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
- Precision-Recall AUC  
- Threshold tuning (best F1-score)

The final model is a **Logistic Regression** classifier with one-hot encoded categorical features and standardized numerical attributes.

---

## 📌 2. Dataset  

Dataset used: **Healthcare Stroke Prediction Dataset**  
File: `healthcare-dataset-stroke-data.csv`

Contains **5110 rows** and **12 columns**.  
The target variable is:
stroke (0 = no stroke, 1 = stroke)

## 📌 3. Project Structure

stroke-risk-prediction/
│
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── train.py # trains the model & saves stroke_model.bin
├── predict.py # API service for inference
├── stroke_model.bin # trained model file
├── healthcare-dataset-stroke-data.csv
└── notebook.ipynb # full EDA + model training + validation


---

## 📌 4. Model Training Pipeline  

### ✔ Steps Performed  

1. **Data Cleaning**  
   - Missing BMI values imputed using median  
   - Stroke column extracted as target  

2. **Train/Validation/Test Split**

3. **Feature Encoding**  
   - One-hot encoding for categorical columns  
   - Numerical columns scaled using `StandardScaler`  

4. **Logistic Regression Model**  
   - Hyperparameter tuning using K-Fold  
   - Best threshold found using validation F1  

5. **Final Model Training**  
   - Model trained on the full training dataset  
   - Exported as `stroke_model.bin` using `pickle`

### Saved Model Contains  
- `dv` → DictVectorizer  
- `scaler` → StandardScaler  
- `model` → Logistic Regression  
- `bmi_median` → For imputing missing BMI  

---

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
### 📸 Screenshot of successful cloud prediction
<img width="954" height="244" alt="Screenshot 2025-11-18 at 6 25 12 AM" src="https://github.com/user-attachments/assets/ba80ec39-8378-4a30-9d79-53891c5be46d" />

