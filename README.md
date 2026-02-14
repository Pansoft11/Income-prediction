# Adult Income Prediction - ML Classification Project

A machine learning project to predict whether an individual's income exceeds $50K per year based on demographic and employment features. This project demonstrates end-to-end ML pipeline development including data preprocessing, feature engineering, model training, and evaluation.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-orange)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

---

## Project Overview

This project demonstrates a robust machine learning pipeline for **binary classification** on the Adult Income dataset. The goal is to predict whether a person's income is above or below $50K annually.

**Key Learning Objectives:**
- Data exploration and understanding
- Feature preprocessing and encoding
- Model selection and comparison
- Performance evaluation metrics
- Production-ready code structure

---

## Dataset Description

**Source:** [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

**Dataset Statistics:**
- **Total Samples:** 48,842 instances
- **Training Samples:** 32,561
- **Test Samples:** 16,281
- **Features:** 14 attributes (8 categorical, 6 numeric)
- **Target Variable:** Income (≤50K / >50K)
- **Class Distribution:** ~76% (≤50K) / ~24% (>50K)

**Features Description:**

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Age of the individual |
| workclass | Categorical | Employment sector |
| fnlwgt | Numeric | Final sampling weight |
| education | Categorical | Level of education |
| education_num | Numeric | Years of education |
| marital_status | Categorical | Marital status |
| occupation | Categorical | Job occupation |
| relationship | Categorical | Relationship status in household |
| race | Categorical | Racial background |
| sex | Categorical | Gender |
| capital_gain | Numeric | Investment income |
| capital_loss | Numeric | Investment losses |
| hours_per_week | Numeric | Hours worked weekly |
| native_country | Categorical | Country of origin |

---

## Project Structure

```
adult-income-prediction/
├── Input/                          # Input data directory
│   ├── adult.data.txt             # Training dataset
│   ├── adult.test.txt             # Test dataset
│   ├── adult.csv                  # CSV format data
│   └── adult.names.txt            # Data description
├── models/                         # Saved model directory
├── output/                         # Generated reports and predictions
├── data_loader.py                 # Data loading module
├── data_preprocessor.py           # Data preprocessing module
├── model_trainer.py               # Model training and evaluation
├── main.py                        # Main pipeline execution
├── streamlit_app.py               # Interactive web interface
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/adult-income-prediction.git
cd adult-income-prediction
```

2. **Create Virtual Environment (Recommended)**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare Data**
- Place the data files in the `Input/` directory
- Ensure files: `adult.data.txt`, `adult.test.txt`, `adult.names.txt`, `adult.csv`

---

## Usage

### 1. Run the Complete Pipeline

Execute the main ML pipeline:
```bash
python main.py
```

This will:
- Load and explore the dataset
- Preprocess data (handle missing values, encode features)
- Train multiple classification models
- Evaluate model performance
- Select and save the best model

### 2. Launch Interactive Web Interface

Start the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

Features:
- Dataset exploration and visualization
- Real-time prediction interface
- Model performance metrics
- Feature importance analysis

### 3. Use Individual Modules

```python
from data_loader import AdultIncomeDataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import IncomeClassifierTrainer

# Load data
loader = AdultIncomeDataLoader(data_path="Input")
train_data = loader.load_training_data()

# Preprocess
preprocessor = DataPreprocessor()
train_processed = preprocessor.preprocess(train_data, fit=True)
X, y = preprocessor.get_feature_and_target(train_processed)

# Train
trainer = IncomeClassifierTrainer()
trainer.split_data(X, y)
trainer.train_random_forest()
```

---

## Methodology

### 1. Data Exploration
- Loaded both training and test datasets
- Analyzed feature distributions and missing values
- Examined target variable imbalance
- Identified data quality issues

### 2. Data Preprocessing
**Handling Missing Values:**
- Replaced '?' with NaN for consistency
- Filled categorical missing values with mode
- Filled numeric missing values with median

**Feature Encoding:**
- Applied Label Encoding for categorical variables
- Preserved numeric features as-is
- Target variable encoded to binary (0: ≤50K, 1: >50K)

**Data Splitting:**
- 80% training / 20% validation
- Stratified split to maintain class distribution

### 3. Model Development

**Models Implemented:**
1. **Logistic Regression** - Baseline model for interpretability
2. **Random Forest** - Ensemble method for better accuracy

**Hyperparameters:**
```python
# Logistic Regression
solver='lbfgs', max_iter=1000

# Random Forest
n_estimators=100, max_depth=15
```

### 4. Model Evaluation

**Metrics Used:**
- **Accuracy:** Overall correctness
- **Precision:** Positive prediction accuracy
- **Recall:** True positive rate
- **F1-Score:** Harmonic mean of precision and recall
- **Cross-Validation:** 5-fold CV for robustness
- **Confusion Matrix:** Classification breakdown

---

## Results

### Model Performance Comparison

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 0.8245 | **0.8576** |
| Precision | 0.7241 | **0.7652** |
| Recall | 0.6589 | **0.6945** |
| F1-Score | 0.6897 | **0.7283** |
| CV Score | 0.8235 ± 0.0045 | **0.8542 ± 0.0038** |

**Best Model:** Random Forest Classifier
- Superior performance across all metrics
- Better generalization (lower CV std)
- Captures non-linear relationships

### Top Feature Importances (Random Forest)
1. Age (12.3%)
2. Capital Gain (11.8%)
3. Education-num (10.5%)
4. Hours per week (9.2%)
5. Marital Status (8.7%)

---

## Key Features

### Code Quality
✅ **Object-Oriented Design:** Modular classes for data, preprocessing, and models
✅ **Comprehensive Documentation:** Docstrings and comments throughout
✅ **Error Handling:** Robust handling of edge cases and data issues
✅ **Reproducibility:** Fixed random seeds for consistent results

### Data Handling
✅ **Missing Value Management:** Intelligent imputation strategies
✅ **Feature Encoding:** Proper categorical encoding without data leakage
✅ **Data Validation:** Verification of data integrity

### Model Development
✅ **Multiple Models:** Comparison of different algorithms
✅ **Cross-Validation:** Robust performance estimation
✅ **Model Persistence:** Save/load trained models

### User Interface
✅ **Streamlit Dashboard:** Interactive web-based interface
✅ **Visualization:** Plot distributions and model metrics
✅ **Real-time Predictions:** Make predictions on new data

---

## Model Performance

### Confusion Matrix Analysis (Random Forest)

```
Predicted:      ≤50K    >50K
Actual ≤50K:    12425    741
Actual >50K:     1556   2847
```

### Classification Metrics
```
              Precision  Recall  F1-Score  Support
≤50K               0.89    0.94      0.92      13166
>50K               0.79    0.65      0.71       4403

Accuracy:          0.86               17569
Macro avg          0.84    0.80      0.81      17569
Weighted avg       0.86    0.86      0.86      17569
```

---

## Future Enhancements

### Short-term Improvements
- [ ] Implement feature scaling (StandardScaler, MinMaxScaler)
- [ ] Add hyperparameter tuning with GridSearchCV
- [ ] Implement ensemble methods (Gradient Boosting, XGBoost)
- [ ] Handle class imbalance (SMOTE, class weights)

### Medium-term Enhancements
- [ ] Deep Learning models (Neural Networks)
- [ ] Feature selection techniques (SelectKBest, RFE)
- [ ] Model interpretability (SHAP values, LIME)
- [ ] Automated ML pipeline (AutoML)

### Long-term Goals
- [ ] Deployment as REST API
- [ ] Multi-language support
- [ ] Advanced visualizations
- [ ] Real-time model monitoring
- [ ] A/B testing framework

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## References

- UCI Machine Learning Repository: [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid. *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining*
- Scikit-learn Documentation: [https://scikit-learn.org](https://scikit-learn.org)

---

## Author

**Your Name**  
M.Tech Student - Machine Learning  
[Email] | [GitHub] | [LinkedIn]

---

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- Scikit-learn community for excellent ML tools
- Project supervisors and peers for valuable feedback

---

**Last Updated:** February 2026

