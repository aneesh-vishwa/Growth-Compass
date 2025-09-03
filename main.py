from src.data_processing import load_and_prepare_data
from src.train import train_and_save_model
from src.evaluate import evaluate_model
from src import config

def main():
    """
    Main function to run the end-to-end uplift modeling pipeline.
    """
    # Step 1: Load and process the data
    X_train, X_test, T_train, T_test, Y_train, Y_test = load_and_prepare_data(config.DATA_PATH)

    # Step 2: Train the uplift model
    model = train_and_save_model(X_train, T_train, Y_train)

    # Step 3: Evaluate the model on the test set
    evaluate_model(model, X_test, T_test, Y_test)

    print("\nUplift modeling pipeline finished successfully!")
    print("Next steps: Use the trained model to score new customers and create a targeted marketing strategy.")


if __name__ == "__main__":
    main()
