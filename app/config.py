# app/config.py

# Define the base directory where the data files are stored
DATA_PATH = "/Users/dianaterraza/Desktop/Data/"

# File paths for each dataset
FILE_PATHS = {
    "stores": f"{DATA_PATH}stores.csv",
    "items": f"{DATA_PATH}items.csv",
    "transactions": f"{DATA_PATH}transactions.csv",
    "oil": f"{DATA_PATH}oil.csv",
    "holidays_events": f"{DATA_PATH}holidays_events.csv",
    "train": f"{DATA_PATH}train.csv",
    "test": f"{DATA_PATH}test.csv"  # Add test if needed
}

# Define the path for the model directory
MODEL_PATH = "/Users/dianaterraza/Desktop/Notebooks_nuevos/model/"

# File path for the xgboost model
MODEL_FILES = {
    "xgboost_model": f"{MODEL_PATH}model.xgb"  # Update the filename if necessary
}
