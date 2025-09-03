# predict.py

import pandas as pd
import joblib
import os
import argparse
from src.data_processing import preprocess_for_prediction
from src import config

def score_customers(input_path, output_path):
    """
    Loads a list of new customers, preprocesses the data, scores them using
    the pre-trained uplift model, and saves the results.
    """
    print(f"Loading new customer data from {input_path}...")
    try:
        new_customers_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return

    # Load the trained models and the training columns
    print("Loading pre-trained models and training columns...")
    try:
        models = joblib.load(config.TWO_MODELS_PATH)
        training_columns = pd.read_csv(f"{config.MODEL_DIR}training_columns.csv").iloc[:, 0].tolist()
    except FileNotFoundError:
        print("Error: Model files not found. Please run main.py to train the model first.")
        return

    # Preprocess the new data
    X_new = preprocess_for_prediction(new_customers_df, training_columns)

    # Score the customers
    print("Scoring customers...")
    prob_treat = models['treatment'].predict_proba(X_new)[:, 1]
    prob_control = models['control'].predict_proba(X_new)[:, 1]
    uplift_scores = prob_treat - prob_control

    # Add scores to the original dataframe
    new_customers_df['uplift_score'] = uplift_scores
    
    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Save the results
    new_customers_df.to_csv(output_path, index=False)
    print(f"Scored customer data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score new customers for uplift.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file with new customer data.")
    parser.add_argument("output_file", type=str, help="Path to save the output CSV file with uplift scores.")
    args = parser.parse_args()

    score_customers(args.input_file, args.output_file)