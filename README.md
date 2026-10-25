# Heart Disease Prediction System

A machine learning web application that predicts heart disease risk using clinical patient data. Built with **4 ML algorithms** (Logistic Regression, Random Forest, KNN, SVM), trained on 918 real patient records from 5 hospital sources, and deployed as a Flask web app on Render.

> **Live Demo**: [heart-disease-predictor-k9jr.onrender.com](https://heart-disease-predictor-k9jr.onrender.com)

---

## Problem Statement

Heart disease is the leading cause of death globally, responsible for approximately 17.9 million deaths per year (WHO). Early detection through clinical indicators can significantly improve patient outcomes. This project builds a predictive system that takes 11 clinical features as input and outputs a binary risk assessment (high risk / low risk) with probability scores.

---

## Dataset

- **Source**: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/amirmahdiabbootalebi/heart-disease) (Kaggle)
- **Origin**: Combined from 5 independent heart disease databases (Cleveland, Hungarian, Switzerland, Long Beach VA, Stalog)
- **Records**: 918 patients | **Features**: 11 clinical attributes | **Target**: Binary (HeartDisease: 0 or 1)
- **Class Distribution**: 508 positive (55.3%) / 410 negative (44.7%)

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Patient age in years |
| Sex | Categorical | M / F |
| ChestPainType | Categorical | ASY, ATA, NAP, TA |
| RestingBP | Numeric | Resting blood pressure (mm Hg) |
| Cholesterol | Numeric | Serum cholesterol (mg/dl) |
| FastingBS | Binary | Fasting blood sugar > 120 mg/dl |
| RestingECG | Categorical | Normal, ST, LVH |
| MaxHR | Numeric | Maximum heart rate achieved |
| ExerciseAngina | Binary | Exercise-induced angina (Y/N) |
| Oldpeak | Numeric | ST depression (numeric) |
| ST_Slope | Categorical | Up, Flat, Down |

---

## Approach

### Data Cleaning
- **172 records** had Cholesterol = 0 (missing values, not biological zeros) - replaced with median
- **1 record** had RestingBP = 0 - replaced with median
- No duplicate rows removed; all 918 records retained

### Preprocessing Pipeline
1. **One-hot encoding** with `drop_first=True` for categorical features (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope) to avoid multicollinearity
2. **Stratified train-test split** (80/20) preserving class ratio in both sets
3. **Standard scaling** fitted on training data only to prevent data leakage

### Models Trained

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | **88.04%** | 86.14% | **90.20%** | 88.12% | 0.930 |
| Random Forest | 85.87% | 85.71% | 86.27% | 85.99% | 0.927 |
| K-Nearest Neighbors | 88.04% | 86.41% | 90.20% | 88.27% | 0.932 |
| Support Vector Machine | 85.33% | 82.35% | 89.22% | 85.64% | **0.942** |

### Why Logistic Regression Was Selected for Deployment
- Highest recall (90.20%) - in medical diagnostics, catching true positives matters more than overall accuracy
- Outputs calibrated probabilities directly via sigmoid function (not approximated)
- Fully interpretable - each coefficient maps to a feature's contribution
- Lightweight and fast inference for web deployment
- Comparable accuracy to more complex models on this dataset

### Key EDA Findings
- **ST_Slope** is the strongest predictor (importance: 0.1708) - an upward slope during exercise indicates healthy heart response
- **MaxHR** (0.1206) and **Oldpeak** (0.1069) are the next most important features
- Asymptomatic chest pain (ASY) has the highest association with heart disease (counterintuitive but clinically documented)
- Optimal K value for KNN: K=6 (90.22% accuracy via elbow method)

---

## Project Structure

```
heart-disease-project/
├── app.py                    # Flask backend - model loading, routes, prediction logic
├── heart_disease_model.pkl   # Serialized Logistic Regression model (joblib)
├── scaler.pkl                # Serialized StandardScaler (joblib)
├── requirements.txt          # Python dependencies
├── templates/
│   ├── index.html            # Prediction form + result display (Jinja2)
│   └── findings.html         # EDA visualizations and model comparison
└── static/
    └── plots/                # EDA plot images (6 PNGs)
        ├── 01_distribution.png
        ├── 02_correlation.png
        ├── 03_chestpain.png
        ├── 04_age.png
        ├── 05_importance.png
        └── 06_roc.png
```

> The full training notebook (with all 4 models, EDA, and evaluation) was developed in Google Colab and is available upon request.

---

## How It Works

1. User enters 11 clinical values through the web form
2. Flask backend receives the POST request and constructs a DataFrame
3. Categorical features are one-hot encoded (matching the training schema)
4. Missing dummy columns are added as 0 (handles the single-patient encoding gap)
5. Features are reordered to match `scaler.feature_names_in_` and scaled
6. The trained Logistic Regression model predicts class (0/1) and probability
7. Result is rendered with risk percentage and confidence score

---

## Run Locally

```bash
# Clone the repository
git clone https://github.com/aashika482/heart-disease-predictor.git
cd heart-disease-predictor

# Install dependencies
pip install -r requirements.txt

# Run the Flask server
python app.py
```

Visit `http://localhost:5000` for the predictor and `http://localhost:5000/findings` for EDA results.

---

## Deployment

Deployed on **Render** using Gunicorn as the WSGI server.

```
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

---

## Tech Stack

- **ML**: scikit-learn, pandas, NumPy
- **Visualization**: matplotlib, seaborn
- **Backend**: Flask, Jinja2
- **Model Serialization**: joblib
- **Deployment**: Render, Gunicorn
- **Development**: Google Colab (training), VS Code (web app)

---

## Training Notebook

The full training pipeline was developed in Google Colab and covers:
- Data loading and exploratory analysis (7 visualizations)
- Data cleaning (handling zero-value anomalies in Cholesterol and RestingBP)
- Preprocessing with one-hot encoding and standard scaling
- Training and evaluation of all 4 models with classification reports
- Confusion matrices, ROC curves, and AUC comparison
- Feature importance analysis (Random Forest)
- KNN hyperparameter tuning (K-value sweep from K=1 to K=20)
- Final model selection rationale and export with joblib

---

## References

1. Heart Failure Prediction Dataset - [Kaggle](https://www.kaggle.com/datasets/amirmahdiabbootalebi/heart-disease)
2. UCI Machine Learning Repository - Heart Disease Dataset
3. Fedesoriano (2021). Heart Failure Prediction Dataset. Kaggle.
4. WHO Cardiovascular Diseases Fact Sheet - [who.int](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))

---

*Built by Aashika M | B.Tech CSE, Manipal University Jaipur*
