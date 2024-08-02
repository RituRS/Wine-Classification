import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Load dataset
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Load pre-trained models
models = {
    'Random Forest': joblib.load('Random Forest_model.pkl'),
    'Logistic Regression': joblib.load('Logistic Regression_model.pkl'),
    'Support Vector Machine': joblib.load('Support Vector Machine_model.pkl')
}

# Map numeric labels to class names
target_names = wine.target_names

# Sidebar for model selection
st.sidebar.header('1. Select a Model')
selected_model = st.sidebar.selectbox('Choose a model to use:', list(models.keys()))

# Get the selected model
model = models[selected_model]

# Sidebar for user input
st.sidebar.header('2. Input Parameters')
st.sidebar.write("Adjust the sliders below to input wine characteristics. These values will be used by the selected model to make a prediction regarding the wine class.")

def user_input_features():
    features = {
        'alcohol': st.sidebar.slider('Alcohol', float(df['alcohol'].min()), float(df['alcohol'].max()), float(df['alcohol'].mean())),
        'malic_acid': st.sidebar.slider('Malic acid', float(df['malic_acid'].min()), float(df['malic_acid'].max()), float(df['malic_acid'].mean())),
        'ash': st.sidebar.slider('Ash', float(df['ash'].min()), float(df['ash'].max()), float(df['ash'].mean())),
        'alcalinity_of_ash': st.sidebar.slider('Alcalinity of ash', float(df['alcalinity_of_ash'].min()), float(df['alcalinity_of_ash'].max()), float(df['alcalinity_of_ash'].mean())),
        'magnesium': st.sidebar.slider('Magnesium', float(df['magnesium'].min()), float(df['magnesium'].max()), float(df['magnesium'].mean())),
        'total_phenols': st.sidebar.slider('Total phenols', float(df['total_phenols'].min()), float(df['total_phenols'].max()), float(df['total_phenols'].mean())),
        'flavanoids': st.sidebar.slider('Flavanoids', float(df['flavanoids'].min()), float(df['flavanoids'].max()), float(df['flavanoids'].mean())),
        'nonflavanoid_phenols': st.sidebar.slider('Nonflavanoid phenols', float(df['nonflavanoid_phenols'].min()), float(df['nonflavanoid_phenols'].max()), float(df['nonflavanoid_phenols'].mean())),
        'proanthocyanins': st.sidebar.slider('Proanthocyanins', float(df['proanthocyanins'].min()), float(df['proanthocyanins'].max()), float(df['proanthocyanins'].mean())),
        'color_intensity': st.sidebar.slider('Color intensity', float(df['color_intensity'].min()), float(df['color_intensity'].max()), float(df['color_intensity'].mean())),
        'hue': st.sidebar.slider('Hue', float(df['hue'].min()), float(df['hue'].max()), float(df['hue'].mean())),
        'od280/od315_of_diluted_wines': st.sidebar.slider('OD280/OD315 of diluted wines', float(df['od280/od315_of_diluted_wines'].min()), float(df['od280/od315_of_diluted_wines'].max()), float(df['od280/od315_of_diluted_wines'].mean())),
        'proline': st.sidebar.slider('Proline', float(df['proline'].min()), float(df['proline'].max()), float(df['proline'].mean()))
    }
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# Make predictions
prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0]
max_prob = prob[prediction]  # Confidence score for the predicted class

# Main interface
st.title('🍷 Wine Classification')
st.write("### Welcome to the Wine Classification Web-App!")
st.write("This web-based application allows you to select a model, input wine features, and view predictions and visualizations.")

st.write("### 1. Model Selection")
st.write(f"You have selected the **{selected_model}** model.")

# Display Results
st.write("### 2. Prediction Results")
st.write(f"**Predicted Class**: {target_names[prediction]}")
st.write(f"**Prediction Probability for {target_names[prediction]}**: {max_prob:.4f}")
st.write(f"**All Class Probabilities**: {dict(zip(target_names, prob))}")

# Display user input parameters
st.write("### 3. User Input Parameters")
st.write("Here are the values you provided:")
st.write(input_df)

# Visualization options
st.write("### 4. Visualizations")
visualization = st.selectbox('Select visualization from the dropdown menu', [
    'Feature Distributions', 
    'Feature Correlation Heatmap',
    'Class Distribution',
    'Confusion Matrix',
    'ROC Curve',
    'Feature Importance',
    'Prediction vs Actual'
])

# Feature Distributions
if visualization == 'Feature Distributions':
    st.write('#### Feature Distributions')
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 15))
    fig.tight_layout(pad=5.0)
    for i, feature in enumerate(df.columns[:-1]):  # Exclude the target column
        sns.histplot(df[feature], ax=axs[i // 4, i % 4], kde=True)
        axs[i // 4, i % 4].set_title(feature)
    plt.show()
    st.pyplot(fig)

# Feature Correlation Heatmap
elif visualization == 'Feature Correlation Heatmap':
    st.write('#### Feature Correlation Heatmap')
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Feature Correlation Heatmap')
    plt.show()
    st.pyplot(fig)

# Class Distribution
elif visualization == 'Class Distribution':
    st.write('#### Class Distribution')
    fig, ax = plt.subplots()
    sns.countplot(x='target', data=df, ax=ax)
    ax.set_xticklabels(target_names)
    ax.set_title('Class Distribution')
    plt.show()
    st.pyplot(fig)

# Confusion Matrix
elif visualization == 'Confusion Matrix':
    st.write('#### Confusion Matrix')
    y_pred = model.predict(df[wine.feature_names])
    cm = confusion_matrix(df['target'], y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    plt.title(f'{selected_model} - Confusion Matrix')
    plt.show()
    st.pyplot(fig)

# ROC Curve
elif visualization == 'ROC Curve':
    st.write('#### ROC Curves')
    y_prob = model.predict_proba(df[wine.feature_names])
    for i in range(len(target_names)):
        fpr, tpr, _ = roc_curve(df['target'], y_prob[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'{target_names[i]} (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.legend(loc='lower right')
        plt.show()
        st.pyplot(fig)

# Feature Importance (For Tree-Based Models)
elif visualization == 'Feature Importance':
    st.write('#### Feature Importance')
    if selected_model == 'Random Forest':
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            fig, ax = plt.subplots()
            sns.barplot(x=importances[indices], y=df.columns[:-1][indices], ax=ax)
            ax.set_title('Feature Importance')
            plt.show()
            st.pyplot(fig)
        else:
            st.write("Feature importance is not available for the selected model.")
    else:
        st.write(f"Feature importance is not applicable for the {selected_model} model as it is not a tree-based model.")

# Prediction vs Actual Comparison
elif visualization == 'Prediction vs Actual':
    st.write('#### Prediction vs Actual Comparison')
    y_pred = model.predict(df[wine.feature_names])
    fig, ax = plt.subplots()
    ax.scatter(df['target'], y_pred, alpha=0.5)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{selected_model} - Prediction vs Actual')
    plt.show()
    st.pyplot(fig)

st.write("### About")
st.write(
    "This web app provides a user-friendly interface to explore and interact with various classification models "
    "using the UCI Wine dataset. The UCI Wine dataset, sourced from the UCI Machine Learning Repository, includes "
    "chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. "
    "The dataset contains features such as alcohol content, malic acid, and total phenols, among others, to classify "
    "the wines into one of three classes."
)