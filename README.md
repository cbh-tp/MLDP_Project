# Telco Customer Churn Predictor

## Project Overview
Customer Churn (cancellation) is one of the biggest cost drivers in the Telco industry. It is significantly more expensive to acquire a new customer than to retain an existing one.

This project builds a **Hybrid Machine Learning Solution** to identify at-risk customers before they leave. Unlike standard models that chase "Accuracy" (and miss churners due to class imbalance), this solution prioritizes **Recall (Class 1)** to minimize "False Negatives"—ensuring we catch as many churners as possible.

## Key Features
* **Hybrid Ensemble Architecture:** Combines **Oversampling** (Voting Classifier) to learn churn patterns with **Undersampling** (Bagging Ensemble) to reduce noise and variance.
* **Business-Centric Evaluation:** Optimized for **Recall (~70%)** rather than just Accuracy, aligning with the business goal of revenue protection.
* **Hyperparameter Tuning:** Utilizes `RandomizedSearchCV` to scientifically optimize the Random Forest component (`n_estimators`, `max_depth`, etc.).
* **Interactive Web App:** A user-friendly Streamlit interface for support agents to calculate churn risk in real-time.
* **Input Validation:** Robust error handling and logic checks (e.g., verifying realistic monthly charges).

## Tech Stack
* **Python 3.9+**
* **Scikit-Learn:** (Logistic Regression, Random Forest, Gradient Boosting, VotingClassifier)
* **Streamlit:** Web Application Framework
* **Pandas & NumPy:** Data Manipulation
* **Matplotlib/Seaborn:** Exploratory Data Analysis
