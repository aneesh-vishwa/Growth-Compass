# src/data_processing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from src import config

def preprocess_for_prediction(df, all_training_columns):
    """
    Preprocesses new, unseen data for prediction, ensuring it has the exact
    same columns (in the same order) as the training data.
    
    This is a more robust implementation to prevent feature mismatch errors.
    """
    # 1. Handle categorical features on the new data
    df_processed = pd.get_dummies(df, columns=config.CATEGORICAL_FEATURES, drop_first=True)

    # 2. Create a clean DataFrame with the correct columns and order, filled with zeros
    df_aligned = pd.DataFrame(columns=all_training_columns)
    df_aligned = pd.concat([df_aligned, df_processed], axis=0, ignore_index=True, sort=False).fillna(0)

    # 3. Ensure all columns are in the final DataFrame, and in the correct order
    return df_aligned[all_training_columns]


def load_and_prepare_data(path):
    """
    Loads and prepares the raw training data.
    """
    df = pd.read_csv(path)
    df_model = df[df['segment'] != config.EXCLUDE_SEGMENT].copy()

    df_model[config.TREATMENT] = df_model['segment'].apply(
        lambda x: 1 if x == config.CAMPAIGN_SEGMENT else 0
    )
    df_model['conversion'] = df_model[config.OUTCOME]

    df_processed = pd.get_dummies(df_model, columns=config.CATEGORICAL_FEATURES, drop_first=True)

    # Define feature set (X), treatment (T), and outcome (Y)
    initial_features = config.FEATURES
    ohe_features = [col for col in df_processed.columns if any(cat in col for cat in config.CATEGORICAL_FEATURES)]
    final_features = initial_features + ohe_features
    
    final_features = [f for f in final_features if f not in ['conversion', config.TREATMENT, config.OUTCOME, 'segment']]

    X = df_processed[final_features]
    T = df_processed[config.TREATMENT]
    Y = df_processed['conversion']

    # --- CRITICAL STEP ---
    # Save the final, ordered list of columns for prediction later
    pd.DataFrame(X.columns, columns=['feature']).to_csv(f"{config.MODEL_DIR}training_columns.csv", index=False)

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.3, random_state=42, stratify=T
    )

    print("Data loaded and prepared successfully.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    return X_train, X_test, T_train, T_test, Y_train, Y_test