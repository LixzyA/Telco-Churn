# Telco Churn Prediction

This repository contains code and resources for predicting customer churn in a telecommunications company.

## Project Structure

- `data/`: Contains raw and processed datasets.
- `models/`: Stores trained machine learning models.
- `src/`: Source code for data processing, feature engineering, and model training.
  - `src/data/`: Scripts for data ingestion and cleaning.
  - `src/features/`: Scripts for feature engineering.
  - `src/notebook/`: Jupyter notebooks for exploratory data analysis and model experimentation.
  - `src/training/`: Scripts for training different machine learning models.

## Models

This project utilizes the following machine learning models:
- PyTorch
- RandomForest
- XGBoost

## Getting Started

### Installation

To set up the project, install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Data Preparation

The `src/data/` directory contains scripts for data ingestion and initial cleaning. The `src/features/main.py` script handles feature engineering, transforming raw data into a format suitable for model training.

### Training

Model training scripts are located in the `src/training/` directory:
- `Pytorch.py`: For training the PyTorch model.
- `RandomForest.py`: For training the RandomForest model.
- `XGBoost.py`: For training the XGBoost model.

## Results

The XGBoost model achieved the best precision score for churn prediction. Below are the detailed classification report metrics:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.77      | 0.98   | 0.87     | 1033    |
| 1     | 0.82      | 0.21   | 0.34     | 374     |
| **Accuracy** |           |        | **0.78** | **1407**|
| **Macro Avg**| 0.80      | 0.60   | 0.60     | 1407    |
| **Weighted Avg**| 0.79      | 0.78   | 0.73     | 1407    |
