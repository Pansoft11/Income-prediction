"""
Data Preprocessing Module
This module handles data cleaning, feature engineering, and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    """
    A class to preprocess the Adult Income dataset including:
    - Handling missing values
    - Feature encoding
    - Feature scaling
    """
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.label_encoders = {}
        self.numeric_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 
                                'capital_loss', 'hours_per_week']
        self.categorical_features = ['workclass', 'education', 'marital_status', 
                                    'occupation', 'relationship', 'race', 
                                    'sex', 'native_country']
        
    def handle_missing_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values (represented as '?') in the dataset
        
        Args:
            dataset: Input DataFrame with missing values
            
        Returns:
            Cleaned DataFrame
        """
        dataset_clean = dataset.copy()
        
        # Replace '?' with NaN for easier handling
        dataset_clean.replace('?', np.nan, inplace=True)
        
        # For categorical columns, fill with mode
        for col in self.categorical_features:
            if col in dataset_clean.columns:
                dataset_clean[col].fillna(dataset_clean[col].mode()[0], inplace=True)
        
        # For numeric columns, fill with median
        for col in self.numeric_features:
            if col in dataset_clean.columns:
                dataset_clean[col].fillna(dataset_clean[col].median(), inplace=True)
        
        return dataset_clean
    
    def encode_categorical_features(self, dataset: pd.DataFrame, 
                                   fit: bool = False) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding
        
        Args:
            dataset: Input DataFrame
            fit: If True, fit the encoders on this data
            
        Returns:
            DataFrame with encoded categorical features
        """
        dataset_encoded = dataset.copy()
        
        for col in self.categorical_features:
            if col not in dataset_encoded.columns:
                continue
                
            if fit:
                # Create and fit new encoder
                encoder = LabelEncoder()
                dataset_encoded[col] = encoder.fit_transform(dataset_encoded[col].astype(str))
                self.label_encoders[col] = encoder
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    # Handle unknown categories by assigning them to the first category
                    dataset_encoded[col] = dataset_encoded[col].map(
                        lambda x: encoder.transform([x])[0] 
                        if x in encoder.classes_ 
                        else 0
                    )
        
        return dataset_encoded
    
    def encode_target_variable(self, dataset: pd.DataFrame, 
                             fit: bool = False) -> pd.DataFrame:
        """
        Encode the target variable (income)
        
        Args:
            dataset: Input DataFrame
            fit: If True, fit the encoder on this data
            
        Returns:
            DataFrame with encoded target variable
        """
        dataset_target = dataset.copy()
        
        if fit:
            encoder = LabelEncoder()
            dataset_target['income'] = encoder.fit_transform(dataset_target['income'])
            self.label_encoders['income'] = encoder
        else:
            if 'income' in self.label_encoders:
                encoder = self.label_encoders['income']
                dataset_target['income'] = encoder.transform(dataset_target['income'])
        
        return dataset_target
    
    def preprocess(self, dataset: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            dataset: Input DataFrame
            fit: If True, fit encoders on this data
            
        Returns:
            Fully preprocessed DataFrame
        """
        # Step 1: Handle missing values
        dataset_processed = self.handle_missing_values(dataset)
        
        # Step 2: Encode categorical features
        dataset_processed = self.encode_categorical_features(dataset_processed, fit=fit)
        
        # Step 3: Encode target variable
        dataset_processed = self.encode_target_variable(dataset_processed, fit=fit)
        
        return dataset_processed
    
    def get_feature_and_target(self, dataset: pd.DataFrame):
        """
        Separate features and target variable
        
        Args:
            dataset: Preprocessed DataFrame
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        target_column = 'income'
        X = dataset.drop(columns=[target_column])
        y = dataset[target_column]
        
        return X, y
