# src/train.py

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src import config

def train_and_save_model(X_train, T_train, Y_train):
    """
    Trains an uplift model using the Two-Model Approach and saves the models.

    This method trains two separate classifiers: one for the treatment group
    and one for the control group.

    Args:
        X_train (pd.DataFrame): Training features.
        T_train (pd.Series): Training treatment assignments (0 or 1).
        Y_train (pd.Series): Training outcomes.

    Returns:
        dict: A dictionary containing the trained 'treatment' and 'control' models.
    """
    print("Training with the Two-Model Approach...")

    # 1. Split the training data into treatment and control sets
    # Ensure indices are aligned by converting Series to a DataFrame
    train_df = pd.concat([X_train, T_train, Y_train], axis=1)
    train_df.columns = list(X_train.columns) + ['treatment', 'conversion']

    df_treat = train_df[train_df['treatment'] == 1]
    df_ctrl = train_df[train_df['treatment'] == 0]

    X_treat_train = df_treat.drop(columns=['treatment', 'conversion'])
    Y_treat_train = df_treat['conversion']
    
    X_ctrl_train = df_ctrl.drop(columns=['treatment', 'conversion'])
    Y_ctrl_train = df_ctrl['conversion']

    # 2. Train a model for each group
    print("Training treatment group model...")
    model_treat = RandomForestClassifier(random_state=42, n_jobs=-1)
    model_treat.fit(X_treat_train, Y_treat_train)

    print("Training control group model...")
    model_ctrl = RandomForestClassifier(random_state=42, n_jobs=-1)
    model_ctrl.fit(X_ctrl_train, Y_ctrl_train)

    print("Model training completed.")

    # 3. Package models into a dictionary
    models = {
        'treatment': model_treat,
        'control': model_ctrl
    }

    # 4. Save the models dictionary
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(models, config.TWO_MODELS_PATH)
    print(f"Models saved to {config.TWO_MODELS_PATH}")

    return models