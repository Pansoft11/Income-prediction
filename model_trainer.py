"""
Model Training Module
This module handles model training and evaluation for income prediction.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score, matthews_corrcoef)
import joblib


class IncomeClassifierTrainer:
    """
    A class to train and evaluate machine learning models for income prediction.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.evaluation_results = {}
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                  test_size: float = 0.2) -> None:
        """
        Split data into training and testing sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
    def train_logistic_regression(self) -> LogisticRegression:
        """
        Train a Logistic Regression model
        
        Returns:
            Trained LogisticRegression model
        """
        print("\n" + "="*60)
        print("Training Logistic Regression Model")
        print("="*60)
        
        lr_model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver='lbfgs'
        )
        
        lr_model.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = lr_model
        
        print("Logistic Regression model trained successfully!")
        return lr_model
    
    def train_random_forest(self, n_estimators: int = 100) -> RandomForestClassifier:
        """
        Train a Random Forest model
        
        Args:
            n_estimators: Number of trees in the forest
            
        Returns:
            Trained RandomForestClassifier model
        """
        print("\n" + "="*60)
        print("Training Random Forest Model")
        print("="*60)
        
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=15
        )
        
        rf_model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf_model
        
        print(f"Random Forest model trained successfully!")
        print(f"Feature Importances (Top 10):")
        self._print_feature_importance(rf_model, top_n=10)
        
        return rf_model
    
    def train_decision_tree(self, max_depth: int = 15) -> DecisionTreeClassifier:
        """
        Train a Decision Tree Classifier model
        
        Args:
            max_depth: Maximum depth of the tree
            
        Returns:
            Trained DecisionTreeClassifier model
        """
        print("\n" + "="*60)
        print("Training Decision Tree Classifier Model")
        print("="*60)
        
        dt_model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=self.random_state,
            criterion='gini',
            min_samples_split=5
        )
        
        dt_model.fit(self.X_train, self.y_train)
        self.models['decision_tree'] = dt_model
        
        print("Decision Tree model trained successfully!")
        print(f"Feature Importances (Top 10):")
        self._print_feature_importance(dt_model, top_n=10)
        
        return dt_model
    
    def train_knn(self, n_neighbors: int = 5) -> KNeighborsClassifier:
        """
        Train a K-Nearest Neighbors Classifier model
        
        Args:
            n_neighbors: Number of neighbors to use
            
        Returns:
            Trained KNeighborsClassifier model
        """
        print("\n" + "="*60)
        print("Training K-Nearest Neighbors Classifier Model")
        print("="*60)
        
        knn_model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='uniform',
            n_jobs=-1
        )
        
        knn_model.fit(self.X_train, self.y_train)
        self.models['knn'] = knn_model
        
        print("K-Nearest Neighbors model trained successfully!")
        
        return knn_model
    
    def train_naive_bayes(self) -> GaussianNB:
        """
        Train a Gaussian Naive Bayes Classifier model
        
        Returns:
            Trained GaussianNB model
        """
        print("\n" + "="*60)
        print("Training Gaussian Naive Bayes Classifier Model")
        print("="*60)
        
        nb_model = GaussianNB()
        
        nb_model.fit(self.X_train, self.y_train)
        self.models['naive_bayes'] = nb_model
        
        print("Gaussian Naive Bayes model trained successfully!")
        
        return nb_model
    
    def train_xgboost(self, n_estimators: int = 100) -> XGBClassifier:
        """
        Train an XGBoost Classifier model
        
        Args:
            n_estimators: Number of boosting rounds
            
        Returns:
            Trained XGBClassifier model
        """
        print("\n" + "="*60)
        print("Training XGBoost Classifier Model")
        print("="*60)
        
        xgb_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=7,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        xgb_model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = xgb_model
        
        print("XGBoost model trained successfully!")
        print(f"Feature Importances (Top 10):")
        self._print_feature_importance(xgb_model, top_n=10)
        
        return xgb_model
    
    def _print_feature_importance(self, model: RandomForestClassifier, 
                                top_n: int = 10) -> None:
        """
        Print feature importances from the model
        
        Args:
            model: Trained model
            top_n: Number of top features to display
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        for i, idx in enumerate(indices):
            print(f"{i+1}. Feature {idx}: {importances[idx]:.4f}")
    
    def evaluate_model(self, model_name: str, model) -> dict:
        """
        Evaluate model performance on test set
        
        Args:
            model_name: Name of the model
            model: Trained model object
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate all 6 evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        mcc_score = matthews_corrcoef(self.y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc_score': mcc_score,
            'predictions': y_pred
        }
        
        self.evaluation_results[model_name] = results
        
        # Print all 6 metrics
        print(f"\n6 PRIMARY EVALUATION METRICS:")
        print(f"{'-'*60}")
        print(f"1. Accuracy:                    {accuracy:.4f}")
        print(f"2. AUC Score:                  {auc_score:.4f}")
        print(f"3. Precision:                  {precision:.4f}")
        print(f"4. Recall:                     {recall:.4f}")
        print(f"5. F1-Score:                   {f1:.4f}")
        print(f"6. Matthews Correlation Coeff: {mcc_score:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['<=50K', '>50K']))
        
        return results
    
    def compare_models(self) -> str:
        """
        Compare all trained models and select the best one
        
        Returns:
            Name of the best model
        """
        print(f"\n{'='*60}")
        print("Model Comparison - All Models with 6 Metrics")
        print(f"{'='*60}")
        
        best_f1 = -1
        best_model_name = None
        
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Accuracy:                    {results['accuracy']:.4f}")
            print(f"  AUC Score:                  {results['auc_score']:.4f}")
            print(f"  Precision:                  {results['precision']:.4f}")
            print(f"  Recall:                     {results['recall']:.4f}")
            print(f"  F1-Score:                   {results['f1_score']:.4f}")
            print(f"  Matthews Correlation Coeff: {results['mcc_score']:.4f}")
            
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'AUC Score': results['auc_score'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'MCC Score': results['mcc_score']
            })
            
            if results['f1_score'] > best_f1:
                best_f1 = results['f1_score']
                best_model_name = model_name
        
        # Display comparison table
        print(f"\n{'='*60}")
        print("Comparison Table:")
        print(f"{'='*60}")
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        print(f"\n{'='*60}")
        print(f"Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"Best F1-Score: {best_f1:.4f}")
        print(f"{'='*60}")
        
        self.best_model = self.models[best_model_name]
        return best_model_name
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save trained model to file
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            print(f"Model {model_name} saved to {filepath}")
        else:
            print(f"Model {model_name} not found!")
    
    def load_model(self, filepath: str) -> object:
        """
        Load a saved model
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model object
        """
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
