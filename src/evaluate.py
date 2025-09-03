# src/evaluate.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causalml.metrics import qini_score
from src import config

def custom_plot_uplift_curve(y_true, uplift, treatment, ax=None):
    """
    This is a custom implementation of the uplift curve plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    df = pd.DataFrame({'y_true': y_true, 'uplift': uplift, 'treatment': treatment})
    df_sorted = df.sort_values(by='uplift', ascending=False).reset_index(drop=True)

    df_sorted['cum_treatment'] = df_sorted['treatment'].cumsum()
    df_sorted['cum_control'] = (1 - df_sorted['treatment']).cumsum()
    
    df_sorted['cum_treatment'] = df_sorted['cum_treatment'].replace(0, 1e-6)
    df_sorted['cum_control'] = df_sorted['cum_control'].replace(0, 1e-6)

    df_sorted['cum_y_treatment'] = (df_sorted['y_true'] * df_sorted['treatment']).cumsum()
    df_sorted['cum_y_control'] = (df_sorted['y_true'] * (1 - df_sorted['treatment'])).cumsum()

    n_total = len(df_sorted)
    uplift_at_k = (df_sorted['cum_y_treatment'] / df_sorted['cum_treatment'] - 
                   df_sorted['cum_y_control'] / df_sorted['cum_control']) * (df_sorted.index + 1)

    ax.plot(np.arange(n_total) + 1, uplift_at_k, label='Model', color='b')
    
    overall_uplift = (df['y_true'][df['treatment'] == 1].mean() - df['y_true'][df['treatment'] == 0].mean())
    ax.plot([0, n_total], [0, overall_uplift * n_total], label='Random', color='k', linestyle='--')
    
    ax.set_xlabel('Number of individuals targeted (sorted by uplift score)')
    ax.set_ylabel('Cumulative Uplift (Incremental Conversions)')
    ax.set_title('Uplift Curve', fontsize=16)
    ax.legend()
    ax.grid(True)
    return ax

def evaluate_model(models, X_test, T_test, Y_test):
    """
    Evaluates the trained Two-Model uplift model.
    """
    print("Evaluating Two-Model Approach...")

    prob_treat = models['treatment'].predict_proba(X_test)[:, 1]
    prob_control = models['control'].predict_proba(X_test)[:, 1]
    uplift_scores = prob_treat - prob_control

    results_df = pd.DataFrame({
        'conversion': Y_test,
        'treatment': T_test,
        'uplift_score': uplift_scores
    })

    qini = qini_score(
        df=results_df,
        outcome_col='conversion',
        treatment_col='treatment',
        score_col='uplift_score'
    )
    
    # --- THIS IS THE FIX ---
    # We extract the single numeric value from the Series using .iloc[0]
    qini_value = qini.iloc[0]
    print(f"Qini Score on Test Set: {qini_value:.4f}")
    # --- END OF FIX ---

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    custom_plot_uplift_curve(y_true=Y_test, uplift=uplift_scores, treatment=T_test, ax=ax)
    plt.savefig(config.UPLIFT_CURVE_PATH)
    
    print(f"Uplift curve plot saved to {config.UPLIFT_CURVE_PATH}")