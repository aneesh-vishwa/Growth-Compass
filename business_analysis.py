# business_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from src import config

def analyze_profit(scored_customers_path):
    """
    Analyzes a scored customer list to generate a profit curve and find the
    optimal targeting strategy.
    """
    print(f"Loading scored customer data from {scored_customers_path}...")
    try:
        df = pd.read_csv(scored_customers_path)
    except FileNotFoundError:
        print(f"Error: The file {scored_customers_path} was not found.")
        return

    # Calculate expected profit for each customer if targeted
    df['expected_profit'] = (df['uplift_score'] * config.VALUE_OF_CONVERSION) - config.COST_OF_TREATMENT

    # Sort customers by uplift score to target the best ones first
    df_sorted = df.sort_values(by='uplift_score', ascending=False).reset_index(drop=True)

    # Calculate cumulative profit
    df_sorted['cumulative_profit'] = df_sorted['expected_profit'].cumsum()

    # Find the optimal point to maximize profit
    max_profit_point = df_sorted['cumulative_profit'].idxmax()
    max_profit = df_sorted['cumulative_profit'].max()
    optimal_customers_to_target = max_profit_point + 1

    print("\n--- Business Analysis Results ---")
    print(f"Maximum potential profit: ${max_profit:.2f}")
    print(f"This is achieved by targeting the top {optimal_customers_to_target} customers.")
    print(f"Targeting more than this will result in diminishing returns.")

    # Generate and save the profit curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted.index + 1, df_sorted['cumulative_profit'], marker='.')
    plt.axvline(x=optimal_customers_to_target, color='r', linestyle='--', label=f'Optimal Point ({optimal_customers_to_target} customers)')
    plt.title('Campaign Profit Curve')
    plt.xlabel('Number of Customers Targeted (Sorted by Uplift Score)')
    plt.ylabel('Cumulative Expected Profit ($)')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    plt.savefig(config.PROFIT_CURVE_PATH)
    print(f"\nProfit curve plot saved to {config.PROFIT_CURVE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the business impact of an uplift model.")
    parser.add_argument("scored_file", type=str, help="Path to the scored customers CSV file.")
    args = parser.parse_args()
    
    analyze_profit(args.scored_file)