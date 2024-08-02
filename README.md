# Wine Classification Web App

This repository contains a Streamlit web application for classifying wines based on their chemical composition using various machine learning models. The models are trained on the [UCI Wine dataset](https://archive.ics.uci.edu/ml/datasets/wine). 

## Features

- **Model Selection**: Choose from Random Forest, Logistic Regression, or Support Vector Machine models.
- **User Input**: Input wine characteristics using sliders to make predictions.
- **Visualizations**: View feature distributions, correlation heatmaps, class distributions, confusion matrices, ROC curves, feature importance, and prediction vs actual comparisons.

## Installation

To run this web app locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/RituRS/Wine-Classification.git
    cd wine-classification-web-app
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download pre-trained models**:
    Ensure the following model files are present in the directory:
    - `Random Forest_model.pkl`
    - `Logistic Regression_model.pkl`
    - `Support Vector Machine_model.pkl`

5. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## App Overview

- **Sidebar**: 
  - Select a model for classification.
  - Input parameters for wine features using sliders.

- **Main Interface**:
  - Displays prediction results including the predicted class and prediction probabilities.
  - Provides various visualizations:
    - Feature Distributions
    - Feature Correlation Heatmap
    - Class Distribution
    - Confusion Matrix
    - ROC Curve
    - Feature Importance (for tree-based models)
    - Prediction vs Actual Comparison

## About

This web app provides a user-friendly interface to explore and interact with various classification models on the UCI Wine dataset. The UCI Wine dataset, sourced from the UCI Machine Learning Repository, includes chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The dataset contains features such as alcohol content, malic acid, and total phenols, among others, to classify the wines into one of three classes.

## Requirements

- Python 3.x
- Streamlit
- Pandas
- Joblib
- Scikit-learn
- Matplotlib
- Seaborn
- NumPy

Install the necessary packages using the requirements.txt file using the following command:
    pip install -r requirements.txt

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine)
- Streamlit for creating interactive web apps with ease.


