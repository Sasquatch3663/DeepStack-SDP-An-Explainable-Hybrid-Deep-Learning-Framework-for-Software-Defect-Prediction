"""
train_pipeline.py

Trains final best model using optimal configuration.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.helper_functions import load_object, save_object
from src.models.baseline_models.logistic_regression import prepare_data
from src.models.baseline_models.random_forest import train_random_forest

from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE


def train_final_model():
    df = load_object("data/processed/encoded.pkl")

    # Step 1: prepare data
    X, y = prepare_data(df)

    # Step 2: feature selection
    selector = SelectKBest(f_classif, k=min(300, X.shape[1]))
    X = selector.fit_transform(X, y)

    # Step 3: sampling
    X, y = SMOTE(random_state=42).fit_resample(X, y)

    # Step 4: train best model (example: RF)
    model = train_random_forest(X, y)

    # Step 5: save
    os.makedirs("models", exist_ok=True)
    save_object("models/final_model.pkl", model)
    save_object("models/feature_selector.pkl", selector)

    print("Final model saved!")


if __name__ == "__main__":
    train_final_model()