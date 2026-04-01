from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import *
from src.data_loader import load_data
from src.preprocessing import handle_missing_values, encode_categorical, select_features
from src.model import PerformancePredictor 
from src.evaluate import evaluate
from src.visualise import (
    plot_correlation,
    plot_predictions,
    plot_feature_relationships,
    plot_residuals,
    plot_target_distribution
)

def main():
    # 1. Load data
    df = load_data(path)

    # 2. Preprocessing
    df = handle_missing_values(df)
    df = encode_categorical(df)

    # 3. Visualization (Pre-analysis)
    plot_target_distribution(df, target)
    plot_correlation(df)

    # 4. Feature selection
    X, y = select_features(df, features, target)

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 6. Train model (Using the Class)
    predictor = PerformancePredictor()
    predictor.train(X_train, y_train)

    # 7. Predict
    y_pred = predictor.predict(X_test)

    # 8. Evaluate
    results = evaluate(y_test, y_pred)

    print("\n" + "="*30)
    print("      MODEL PERFORMANCE")
    print("="*30)
    for key, value in results.items():
        print(f"{key:4}: {value:.4f}")
    print("="*30)

    # 9. Visualization (Post-analysis)
    plot_predictions(y_test, y_pred)
    plot_feature_relationships(df, target)
    plot_residuals(y_test, y_pred)

if __name__ == "__main__":
    main()