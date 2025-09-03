# app.py

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.data_processing import preprocess_for_prediction
from src import config

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# --- App Title ---
st.title("🎯 Uplift Modeling for Targeted Marketing")

# --- Load Models and Columns (cached for performance) ---
@st.cache_resource
def load_model_assets():
    try:
        models = joblib.load(config.TWO_MODELS_PATH)
        training_columns = pd.read_csv(f"{config.MODEL_DIR}training_columns.csv").iloc[:, 0].tolist()
        return models, training_columns
    except FileNotFoundError:
        return None, None

models, training_columns = load_model_assets()

if models is None:
    st.error("Model not found! Please run `python main.py` to train the model first.")
else:
    # --- Sidebar for User Input ---
    st.sidebar.header("Campaign Parameters")
    cost_treatment = st.sidebar.number_input(
        "Cost per Promotion ($)", 
        min_value=0.0, 
        value=config.COST_OF_TREATMENT, 
        step=0.01
    )
    value_conversion = st.sidebar.number_input(
        "Value per Conversion ($)", 
        min_value=0.0, 
        value=config.VALUE_OF_CONVERSION, 
        step=0.1
    )
    
    st.sidebar.header("Upload Customer Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file of customers to score", type=["csv"])

    if uploaded_file is not None:
        # --- Process and Score Data ---
        df_new = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Customer Data (First 5 Rows)")
        st.dataframe(df_new.head())

        X_new = preprocess_for_prediction(df_new.copy(), training_columns)

        prob_treat = models['treatment'].predict_proba(X_new)[:, 1]
        prob_control = models['control'].predict_proba(X_new)[:, 1]
        uplift_scores = prob_treat - prob_control

        results_df = df_new.copy()
        results_df['uplift_score'] = uplift_scores
        results_df['expected_profit'] = (results_df['uplift_score'] * value_conversion) - cost_treatment
        results_df_sorted = results_df.sort_values(by='uplift_score', ascending=False).reset_index(drop=True)
        results_df_sorted['cumulative_profit'] = results_df_sorted['expected_profit'].cumsum()

        # --- Find Optimal Targeting ---
        max_profit_point = results_df_sorted['cumulative_profit'].idxmax()
        max_profit = results_df_sorted['cumulative_profit'].max()
        optimal_customers_to_target = max_profit_point + 1

        # --- Display Results ---
        st.subheader("📈 Campaign Profit Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Profit Curve Plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(results_df_sorted.index + 1, results_df_sorted['cumulative_profit'], marker='.')
            ax.axvline(x=optimal_customers_to_target, color='r', linestyle='--', label=f'Optimal Point ({optimal_customers_to_target} customers)')
            ax.set_title('Campaign Profit Curve')
            ax.set_xlabel('Number of Customers Targeted')
            ax.set_ylabel('Cumulative Expected Profit ($)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        with col2:
            st.metric(label="Maximum Potential Profit", value=f"${max_profit:,.2f}")
            st.metric(label="Optimal Customers to Target", value=f"{optimal_customers_to_target}")
            st.info("This is the point where targeting more customers will start to decrease overall campaign profit.")
            
            # Allow downloading the scored results
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df_to_csv(results_df_sorted)
            st.download_button(
                 label="Download Scored Customer List",
                 data=csv,
                 file_name='scored_customers.csv',
                 mime='text/csv',
             )

        # --- Display Top Customers ---
        st.subheader(" Top Customers to Target (Persuadables)")
        st.dataframe(results_df_sorted.head(20))