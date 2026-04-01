# Student Performance Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

## Overview

This project implements a machine learning pipeline to predict a student's Performance Index based on academic and lifestyle factors.

The system uses Multiple Linear Regression and follows a modular, production-style architecture. It also includes an optional Streamlit-based web interface for real-time predictions.

---

## Problem Statement

Predict the Performance Index (final score) using:

- Hours Studied  
- Previous Scores  
- Sleep Hours  
- Sample Question Papers Practiced  
- Extracurricular Activities  

---

## Approach and Design

The system is designed using the following principles:

### Modularity
Each component (data loading, preprocessing, modeling, evaluation) is separated into independent modules to improve maintainability and testing.

### Data Validation
The dataset is validated before processing to ensure schema correctness and prevent runtime failures.

### Consistent Feature Pipeline
Feature names and order are centrally managed to ensure consistency between training and inference.

### Separation of Concerns
Core machine learning logic is separated from the user interface.

---

## Project Structure

student-performance-regression/
│
├── data/
│   └── student_data.csv
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── validator.py
│   ├── model.py
│   ├── evaluate.py
│   ├── visualize.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_validator.py
│   ├── test_pipeline.py
│
├── app/
│   └── streamlit_app.py
│
├── main.py
├── requirements.txt
└── README.md

---

## Installation and Setup

### Clone the repository

git clone https://github.com/your-username/student-performance-regression.git  
cd student-performance-regression  

### Create virtual environment (recommended)

python -m venv venv  

Activate environment:

Windows  
venv\Scripts\activate  

Mac/Linux  
source venv/bin/activate  

### Install dependencies

pip install -r requirements.txt  

### Add dataset

Place the dataset file in:

data/student_data.csv  

---

## Running the Project

### Run training pipeline

python main.py  

### Run Streamlit application (optional)

streamlit run app/streamlit_app.py  

---

## Model Performance

Example results:

MAE: 1.61  
MSE: 4.08  
R2 Score: 0.989  

- The model explains approximately 99% of the variance  
- Prediction error is low, indicating strong performance  

---

## Visualizations

The project includes:

- Correlation Heatmap  
- Feature vs Target Analysis (Boxplots and Density plots)  
- Actual vs Predicted Plot  
- Residual Plot  
- Target Distribution  

These visualizations help interpret model behavior and validate assumptions.

### 📊 Analysis Results

![Figure 1](results/Figure_1.png)
![Figure 2](results/Figure_2.png)
![Figure 3](results/Figure_3.png)
![Figure 4](results/Figure_4.png)
![Figure 5](results/Figure_5.png)
![Figure 6](results/Figure_6.png)
![Figure 7](results/Figure_7.png)
![Figure 8](results/Figure_8.png)
![Figure 9](results/Figure_9.png)

---

## Testing

Run all tests:

pytest  

or:

python -m unittest discover tests  

---

## Features

- Modular and scalable architecture  
- Data validation layer  
- Robust preprocessing pipeline  
- Multiple visualization techniques  
- Unit and integration testing  
- Optional Streamlit interface  

---

## Edge Case Handling

- Missing values handled for numeric and categorical features  
- Invalid categorical values rejected  
- Dataset schema validation implemented  
- Fail-fast behavior for incorrect inputs  

---

## Future Improvements

- Add cross-validation  
- Experiment with Ridge and Lasso regression  
- Save model using joblib or pickle  
- Deploy using Streamlit Cloud  
- Add CI/CD pipeline (GitHub Actions)  
- Implement feature scaling  
