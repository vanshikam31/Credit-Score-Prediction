# Credit Score Prediction using Machine Learning

An end-to-end Machine Learning project that predicts an individual's creditworthiness based on demographic and financial information. The application classifies customers into **High**, **Average**, and **Low** credit score categories and provides real-time predictions through an interactive Streamlit web application.

## Features

* Data Preprocessing and Cleaning
* Label encoding for categorical variables
* Train-test split for model validation
* Implementation of Logistic Regression, Decision Tree, and Random Forest classifiers
* Model evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
* Best model selection based on performance
* Model persistence using Joblib
* Real-time prediction using Streamlit

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib

## Project Structure

```text
credit-score-prediction/
├── app.py
├── credit_score_model.py
├── credit_model.pkl
├── encoders.pkl
├── target_encoder.pkl
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/vanshikam31/credit-score-prediction.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

## Project Workflow

Dataset → Data Preprocessing → Label Encoding → Model Training → Model Evaluation → Best Model Selection → Model Saving → Streamlit Integration → Real-Time Prediction

## Model Used

After comparing multiple classification algorithms, **Random Forest Classifier** was selected as the final model due to its superior performance on the dataset.

## Future Improvements

* Use a larger and more realistic credit dataset
* Add ROC-AUC and feature importance visualizations
* Deploy the application on Streamlit Cloud
* Improve handling of imbalanced datasets using SMOTE

## Author

Developed by **Vanshika Mehra** as a Machine Learning portfolio project.
