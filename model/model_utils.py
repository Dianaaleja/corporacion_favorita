import xgboost as xgb
from app.config import MODEL_PATH

def load_model(model_path=MODEL_PATH):
    """
    Loads a pre-trained XGBoost model from the local filesystem.
    """
    # Define the path to the XGBoost model file
    model_file_path = f"{model_path}model.xgb"

    # Load the XGBoost model from the file
    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model(model_file_path)

    print(f"Model loaded successfully from {model_file_path}")
    return xgboost_model

def predict(model, input_data):
    """
    Runs prediction on input data using the pre-trained model.
    """
    # Drop the original 'date' column and 'unit_sales' column from input data
    input_data = input_data.drop(columns=['date', 'unit_sales'])

    # Run the prediction
    prediction = model.predict(input_data)
    return prediction
