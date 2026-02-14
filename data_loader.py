"""
Data Loading Module
This module handles loading and initial exploration of the Adult Income dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class AdultIncomeDataLoader:
    """
    A class to load and manage the Adult Income dataset from UCI ML repository.
    This dataset contains census income data for income prediction tasks.
    """
    
    def __init__(self, data_path: str = "Input"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the directory containing the data files
        """
        self.data_path = Path(data_path)
        self.training_data = None
        self.test_data = None
        self.feature_names = None
        
    def load_training_data(self) -> pd.DataFrame:
        """
        Load the training dataset (adult.data.txt)
        
        Returns:
            DataFrame containing training data
        """
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
        ]
        
        training_file = self.data_path / 'adult.data.txt'
        self.training_data = pd.read_csv(
            training_file,
            header=None,
            names=column_names,
            skipinitialspace=True
        )
        
        return self.training_data
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load the test dataset (adult.test.txt)
        
        Returns:
            DataFrame containing test data
        """
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
        ]
        
        test_file = self.data_path / 'adult.test.txt'
        self.test_data = pd.read_csv(
            test_file,
            header=None,
            names=column_names,
            skipinitialspace=True,
            skiprows=1  # Skip the first line which contains a note
        )
        
        # Clean income labels (remove periods if present)
        self.test_data['income'] = self.test_data['income'].str.replace('.', '', regex=False)
        
        return self.test_data
    
    def get_dataset_statistics(self) -> dict:
        """
        Get basic statistics about the dataset
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self.training_data is None:
            self.load_training_data()
        
        stats = {
            'training_samples': len(self.training_data),
            'feature_count': len(self.training_data.columns) - 1,  # Excluding target
            'missing_values': self.training_data.isnull().sum().to_dict(),
            'income_distribution': self.training_data['income'].value_counts().to_dict()
        }
        
        return stats
    
    def display_data_info(self) -> None:
        """Display basic information about the loaded data"""
        if self.training_data is None:
            self.load_training_data()
        
        print("=" * 80)
        print("ADULT INCOME DATASET INFORMATION")
        print("=" * 80)
        print(f"\nDataset Shape: {self.training_data.shape}")
        print(f"Features: {self.training_data.columns.tolist()}")
        print("\nData Types:")
        print(self.training_data.dtypes)
        print("\nMissing Values:")
        print(self.training_data.isnull().sum())
        print("\nTarget Variable Distribution:")
        print(self.training_data['income'].value_counts())
        print("=" * 80)
