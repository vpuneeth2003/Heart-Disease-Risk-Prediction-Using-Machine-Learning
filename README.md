# Heart Disease Prediction Project

# ECE 4424 Final Project
## Puneeth Vangumalla
## CRN: 83753
## Dr. Debswapna Bhattacharya

## Overview

This project is a machine learning application that predicts the risk of heart disease based on clinical data. It uses three machine learning models:
- **Logistic Regression**
- **Neural Network**
- **Naive Bayes**

The project evaluates the performance of these models on various metrics and allows users to input clinical data to make predictions in real-time with a confidence score.

---

## Features

- **Train and Evaluate Models**: Trains three machine learning models and evaluates their performance using metrics like accuracy, precision, recall, F1 score, and ROC-AUC.
- **Visualization**: Generates a comparative bar plot for model performance across metrics and plots ROC curves for all models.
- **Real-Time Prediction**: Includes a user-interactive pipeline to input clinical data and predict heart disease risk using the trained Neural Network model.

---

## Dataset

The dataset `heart.csv` is sourced from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/dataset/45/heart+disease). It contains the following fields:
- **age**: Age in years
- **sex**: 1 = male; 0 = female
- **cp**: Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic)
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = probable/definite left ventricular hypertrophy)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise-induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)
- **ca**: Number of major vessels (0–3) colored by fluoroscopy
- **thal**: 3 = normal, 6 = fixed defect, 7 = reversible defect
- **target**: 0 = no heart disease; 1 = heart disease

---

## Files in Repository

- **`HeartDisease_Train&Predict.py`**: Python script containing the entire code for training, evaluation, and prediction.
- **`HeartDisease_Train&Predict.ipynb`**: Jupyter Notebook version of the project for step-by-step exploration and visualization.
- **`heart.csv`**: Dataset containing clinical data for training and evaluation.

---

## How to Use

### 1. Train and Evaluate Models
Run the Python script or Jupyter Notebook to train and evaluate the three models. The script will:
- Train the models on 80% of the dataset.
- Evaluate their performance using test data (20% of the dataset).
- Plot comparative metrics and ROC curves for all models.
- Print a summary of results.

### 2. Make Real-Time Predictions
Using the `predictHeartDisease` function in the script/notebook, the implementation can also provide predictions:
1. Enter clinical data when prompted (13 attributes such as age, cholesterol level, etc.).
2. The trained Neural Network model will predict whether the user is at high or low risk of heart disease.
3. A confidence score will also be displayed to indicate the model's certainty.

---

## Requirements

- Python 3.7 or higher
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## References

1. [H. El-Sofany et al., "A proposed technique for predicting heart disease using machine learning algorithms and an explainable AI method"](https://doi.org/10.1038/s41598-024-74656-2)
2. [UCI Machine Learning Repository: Heart Disease Dataset](http://archive.ics.uci.edu/dataset/45/heart+disease)
3. [D. Lapp, "Heart disease dataset" on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data)
4. [O. Jeremiah, "Building a heart disease prediction model using machine learning"](https://medium.com/@oluseyejeremiah/building-a-heart-disease-prediction-model-using-machine-learning-4c690243a93e)
5. [N. L. Fitriyani et al., "HDPM: An effective heart disease prediction model for a clinical decision support system"](https://doi.org/10.1109/access.2020.3010511)
6. [M. M. Ali et al., "Heart disease prediction using supervised machine learning algorithms: Performance Analysis and comparison"](https://doi.org/10.1016/j.compbiomed.2021.104672)
7. [P. W. Wilson et al., "Prediction of coronary heart disease using risk factor categories"](https://doi.org/10.1161/01.cir.97.18.1837)
8. [GeeksforGeeks, "Understanding the predict_proba() function in Scikit-learn’s SVC"](https://www.geeksforgeeks.org/understanding-the-predictproba-function-in-scikit-learns-svc/)
