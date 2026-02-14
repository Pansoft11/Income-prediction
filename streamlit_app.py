"""
Streamlit Web Application for Income Prediction
An interactive dashboard for data exploration and making predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import AdultIncomeDataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import IncomeClassifierTrainer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        padding: 2rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)


COLUMN_NAMES = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

MODEL_CHOICES = {
    'Logistic Regression': 'logistic_regression',
    'Decision Tree': 'decision_tree',
    'kNN': 'knn',
    'Naive Bayes': 'naive_bayes',
    'Random Forest (Ensemble)': 'random_forest',
    'XGBoost (Ensemble)': 'xgboost'
}


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and clean string fields"""
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    if cleaned.shape[1] == len(COLUMN_NAMES):
        if cleaned.columns.tolist() != COLUMN_NAMES:
            cleaned.columns = COLUMN_NAMES

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    if 'income' in cleaned.columns:
        cleaned['income'] = cleaned['income'].str.replace('.', '', regex=False)

    return cleaned


@st.cache_resource
def load_data_cached():
    """Load data with caching"""
    loader = AdultIncomeDataLoader(data_path="Input")
    train_data = loader.load_training_data()
    return train_data


@st.cache_resource
def load_model_cached():
    """Load trained model with caching"""
    model_path = "models/random_forest.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_resource
def get_preprocessor_and_models(train_df: pd.DataFrame | None = None):
    """Train all models once and reuse them for evaluation"""
    if train_df is None:
        loader = AdultIncomeDataLoader(data_path="Input")
        train_data = loader.load_training_data()
    else:
        train_data = normalize_dataset(train_df)

    preprocessor = DataPreprocessor()
    train_processed = preprocessor.preprocess(train_data, fit=True)
    X_train, y_train = preprocessor.get_feature_and_target(train_processed)

    trainer = IncomeClassifierTrainer()
    trainer.split_data(X_train, y_train, test_size=0.2)
    trainer.train_logistic_regression()
    trainer.train_decision_tree()
    trainer.train_knn()
    trainer.train_naive_bayes()
    trainer.train_random_forest(n_estimators=100)
    trainer.train_xgboost(n_estimators=100)

    return preprocessor, trainer.models


def main():
    """Main Streamlit application"""
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Home", "Data Exploration", "EDA & Visualization", "Make Prediction", "Model Evaluation"]
    )
    
    if page == "Home":
        show_home_page()
    elif page == "Data Exploration":
        show_data_exploration()
    elif page == "EDA & Visualization":
        show_eda_page()
    elif page == "Make Prediction":
        show_prediction_page()
    elif page == "Model Evaluation":
        show_model_info()


def show_home_page():
    """Display home page"""
    st.title("💼 Adult Income Prediction System")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Welcome!")
        st.markdown("""
        ### Project Overview
        This machine learning application predicts whether an individual's annual income 
        exceeds $50K based on demographic and employment features.
        
        ### What You Can Do:
        - **📈 Explore Data:** Analyze the Adult Income dataset in detail
        - **📊 Visualizations:** View distributions and relationships
        - **🔮 Make Predictions:** Get predictions for new individuals
        - **ℹ️ Model Details:** Learn about the trained models
        
        ### Dataset Info:
        - **Source:** UCI Machine Learning Repository
        - **Samples:** 48,842 individuals
        - **Features:** 14 attributes (demographics, employment, financial)
        - **Target:** Income (≤50K or >50K annually)
        """)
    
    with col2:
        st.info("""
        ### Quick Stats
        - 📦 Training Samples: 32,561
        - 🧪 Test Samples: 16,281
        - ⚖️ Class Balance: 76% vs 24%
        - 🎯 Best Model Accuracy: 85.76%
        """)
    
    st.markdown("---")
    st.subheader("🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 Explore Data", use_container_width=True):
            st.switch_page("pages/data_exploration.py")
    with col2:
        if st.button("🔮 Make Prediction", use_container_width=True):
            st.switch_page("pages/prediction.py")
    with col3:
        if st.button("📊 View Models", use_container_width=True):
            st.switch_page("pages/model_info.py")


def show_data_exploration():
    """Display data exploration page"""
    st.title("📊 Data Exploration")
    st.markdown("---")
    
    # Load data
    try:
        data = load_data_cached()
        
        # Display dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", f"{len(data):,}")
        with col2:
            st.metric("Total Features", len(data.columns))
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.markdown("---")
        
        # Display data
        st.subheader("Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Data statistics
        st.subheader("Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
        
        # Data types
        st.subheader("Column Information")
        col1, col2 = st.columns([2, 1])
        with col1:
            dtypes_df = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes,
                'Non-Null': [f"{data[col].notna().sum()}/{len(data)}" for col in data.columns]
            })
            st.dataframe(dtypes_df, use_container_width=True)
        
        with col2:
            st.info("**Data Types**\nNumeric columns for analysis\nCategorical columns for encoding")
        
        # Target variable analysis
        st.subheader("Target Variable Distribution")
        income_dist = data['income'].value_counts()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write(income_dist)
            percentages = (income_dist / len(data) * 100).round(2)
            st.write("**Percentages:**")
            for label, pct in percentages.items():
                st.write(f"{label}: {pct}%")
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#FF6B6B', '#4ECDC4']
            income_dist.plot(kind='bar', ax=ax, color=colors)
            ax.set_title("Income Distribution", fontsize=14, fontweight='bold')
            ax.set_ylabel("Count")
            ax.set_xlabel("Income Level")
            plt.xticks(rotation=0)
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")


def show_eda_page():
    """Display EDA and visualization page"""
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    try:
        data = load_data_cached()
        
        # Visualization options
        st.subheader("Select Visualization Type")
        viz_type = st.selectbox(
            "Choose visualization:",
            ["Age Distribution", "Education Distribution", "Workclass Distribution", 
             "Income vs Age", "Income vs Education"]
        )
        
        col1, col2 = st.columns(2)
        
        if viz_type == "Age Distribution":
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                data['age'].hist(bins=30, ax=ax, color='#3498db', edgecolor='black')
                ax.set_title("Age Distribution", fontsize=14, fontweight='bold')
                ax.set_xlabel("Age")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
            with col2:
                st.info(f"""
                **Age Statistics:**
                - Mean: {data['age'].mean():.2f} years
                - Median: {data['age'].median():.2f} years
                - Std Dev: {data['age'].std():.2f} years
                - Range: {data['age'].min()}-{data['age'].max()} years
                """)
        
        elif viz_type == "Education Distribution":
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                edu_counts = data['education'].value_counts().head(10)
                edu_counts.plot(kind='barh', ax=ax, color='#2ecc71')
                ax.set_title("Top 10 Education Levels", fontsize=14, fontweight='bold')
                ax.set_xlabel("Count")
                st.pyplot(fig)
            with col2:
                st.info(f"""
                **Education Info:**
                - Unique levels: {data['education'].nunique()}
                - Most common: {data['education'].value_counts().index[0]}
                - Count: {data['education'].value_counts().iloc[0]}
                """)
        
        elif viz_type == "Workclass Distribution":
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                work_counts = data['workclass'].value_counts()
                work_counts.plot(kind='barh', ax=ax, color='#e74c3c')
                ax.set_title("Workclass Distribution", fontsize=14, fontweight='bold')
                ax.set_xlabel("Count")
                st.pyplot(fig)
            with col2:
                st.info(f"""
                **Workclass Info:**
                - Unique categories: {data['workclass'].nunique()}
                - Most common: {data['workclass'].value_counts().index[0]}
                - Count: {data['workclass'].value_counts().iloc[0]}
                """)
        
        elif viz_type == "Income vs Age":
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                for income in data['income'].unique():
                    mask = data['income'] == income
                    ax.hist(data[mask]['age'], bins=20, alpha=0.6, label=income)
                ax.set_title("Age Distribution by Income Level", fontsize=14, fontweight='bold')
                ax.set_xlabel("Age")
                ax.set_ylabel("Frequency")
                ax.legend()
                st.pyplot(fig)
            
            with col2:
                for income in data['income'].unique():
                    mask = data['income'] == income
                    st.write(f"**{income}**")
                    st.write(f"Mean age: {data[mask]['age'].mean():.2f}")
                    st.write(f"Median age: {data[mask]['age'].median():.2f}")
        
        elif viz_type == "Income vs Education":
            with col1:
                fig, ax = plt.subplots(figsize=(12, 6))
                crosstab = pd.crosstab(data['education'], data['income'])
                crosstab.plot(kind='bar', ax=ax)
                ax.set_title("Education vs Income", fontsize=14, fontweight='bold')
                ax.set_xlabel("Education Level")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with col2:
                st.dataframe(crosstab, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error during visualization: {str(e)}")


def show_prediction_page():
    """Display prediction form"""
    st.title("🔮 Make Income Prediction")
    st.markdown("---")
    
    # Display remaining code trimmed for brevity...
    st.subheader("Enter Individual Details")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 90, 35)
            education = st.selectbox(
                "Education",
                ['Bachelors', 'HS-grad', 'Some-college', 'Masters', 'Doctorate',
                 'Associates', 'Prof-school', 'Cant-identify', '12th']
            )
            workclass = st.selectbox(
                "Workclass",
                ['Private', 'Self-emp-not-inc', 'Federal-gov', 'State-gov', 'Local-gov',
                 'Self-emp-inc', 'Without-pay', 'Never-worked']
            )
            marital_status = st.selectbox(
                "Marital Status",
                ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
                 'Married-AF-spouse', 'Widowed']
            )
        
        with col2:
            occupation = st.selectbox(
                "Occupation",
                ['Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Sales',
                 'Craft-repair', 'Tech-support', 'Adm-clerical', 'Transport-moving']
            )
            hours_per_week = st.slider("Hours per Week", 1, 100, 40)
            sex = st.selectbox("Sex", ['Male', 'Female'])
            capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        
        submit_button = st.form_submit_button("Make Prediction", use_container_width=True)
    
    if submit_button:
        st.info("✅ Prediction feature requires trained model. Train the model first using main.py")


def show_model_info():
    """Display model evaluation page"""
    st.title("✅ Model Evaluation")
    st.markdown("---")

    st.subheader("Dataset Upload (CSV)")
    st.caption("Upload test data only (CSV format). The file should include an 'income' column.")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

    st.subheader("Model Selection")
    model_label = st.selectbox("Choose a model", list(MODEL_CHOICES.keys()))

    if uploaded_file is None:
        st.info("Please upload a CSV file to evaluate the selected model.")
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {str(e)}")
        return

    raw_df = normalize_dataset(raw_df)

    if 'income' not in raw_df.columns:
        st.error("The uploaded CSV must include the 'income' column for evaluation metrics.")
        return

    with st.spinner("Preparing data and loading models..."):
        try:
            preprocessor, models = get_preprocessor_and_models(raw_df)
        except FileNotFoundError:
            st.error("Training data not found in Input/. Upload a CSV to train models inside the app.")
            return
        model_key = MODEL_CHOICES[model_label]
        model = models.get(model_key)

    if model is None:
        st.error(f"Model '{model_label}' is not available. Train models by running main.py.")
        return

    processed = preprocessor.preprocess(raw_df, fit=False)
    X_eval, y_eval = preprocessor.get_feature_and_target(processed)

    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_eval, y_pred),
        'AUC': roc_auc_score(y_eval, y_proba),
        'Precision': precision_score(y_eval, y_pred),
        'Recall': recall_score(y_eval, y_pred),
        'F1': f1_score(y_eval, y_pred),
        'MCC': matthews_corrcoef(y_eval, y_pred)
    }

    st.subheader("Evaluation Metrics")
    metrics_df = pd.DataFrame([metrics])
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Classification Report")
    report = classification_report(y_eval, y_pred, target_names=['<=50K', '>50K'])
    st.text(report)

    show_cm = st.checkbox("Show Confusion Matrix")
    if show_cm:
        cm = confusion_matrix(y_eval, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)


if __name__ == "__main__":
    main()
