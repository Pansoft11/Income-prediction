"""
Main Pipeline Script
This script runs the complete ML pipeline for income prediction.
"""

import pandas as pd
from data_loader import AdultIncomeDataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import IncomeClassifierTrainer


def main():
    """
    Execute the complete pipeline:
    1. Load data
    2. Preprocess data
    3. Train models
    4. Evaluate models
    5. Select best model
    """
    
    print("="*80)
    print("ADULT INCOME PREDICTION - ML PIPELINE")
    print("="*80)
    
    # Step 1: Load Data
    print("\n[STEP 1] Loading Dataset...")
    print("-" * 80)
    
    loader = AdultIncomeDataLoader(data_path="Input")
    training_data = loader.load_training_data()
    test_data = loader.load_test_data()
    
    loader.display_data_info()
    stats = loader.get_dataset_statistics()
    print(f"\nIncome Distribution in Training Data:")
    for income_level, count in stats['income_distribution'].items():
        percentage = (count / stats['training_samples']) * 100
        print(f"  {income_level}: {count} ({percentage:.2f}%)")
    
    # Step 2: Preprocess Data
    print("\n[STEP 2] Preprocessing Data...")
    print("-" * 80)
    
    preprocessor = DataPreprocessor()
    
    # Preprocess training data
    train_processed = preprocessor.preprocess(training_data, fit=True)
    X_train_full, y_train_full = preprocessor.get_feature_and_target(train_processed)
    
    # Preprocess test data
    test_processed = preprocessor.preprocess(test_data, fit=False)
    X_test_full, y_test_full = preprocessor.get_feature_and_target(test_processed)
    
    print(f"Training data shape after preprocessing: {X_train_full.shape}")
    print(f"Test data shape after preprocessing: {X_test_full.shape}")
    print(f"Missing values in processed training data: {X_train_full.isnull().sum().sum()}")
    print(f"Missing values in processed test data: {X_test_full.isnull().sum().sum()}")
    
    # Step 3: Train Models
    print("\n[STEP 3] Training Models...")
    print("-" * 80)
    
    trainer = IncomeClassifierTrainer()
    
    # Split data into train and validation sets
    trainer.split_data(X_train_full, y_train_full, test_size=0.2)
    
    # Train multiple models
    trainer.train_logistic_regression()
    trainer.train_random_forest(n_estimators=100)
    
    # Step 4: Evaluate Models
    print("\n[STEP 4] Evaluating Models...")
    print("-" * 80)
    
    for model_name, model in trainer.models.items():
        trainer.evaluate_model(model_name, model)
    
    # Step 5: Select Best Model
    print("\n[STEP 5] Model Selection...")
    print("-" * 80)
    
    best_model_name = trainer.compare_models()
    
    # Step 6: Save Best Model
    print("\n[STEP 6] Saving Best Model...")
    print("-" * 80)
    
    best_model = trainer.models[best_model_name]
    trainer.save_model(best_model_name, f"models/{best_model_name}.joblib")
    
    # Step 7: Final Predictions on Unseen Test Data
    print("\n[STEP 7] Making Final Predictions on Test Set...")
    print("-" * 80)
    
    final_predictions = best_model.predict(X_test_full)
    test_accuracy = (final_predictions == y_test_full).mean()
    
    print(f"Final Test Set Accuracy: {test_accuracy:.4f}")
    print(f"Number of predictions: {len(final_predictions)}")
    print(f"Income >50K predictions: {(final_predictions == 1).sum()}")
    print(f"Income <=50K predictions: {(final_predictions == 0).sum()}")
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
