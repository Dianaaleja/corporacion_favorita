
''''
import sys 
import os
project_root = "/Users/main/Desktop/Notebooks_nuevos"
if project_root not in sys.path:
    sys.path.append(project_root)

'''

import streamlit as st
from data.data_utils import load_data, preprocess_input_data
from model.model_utils import load_model, predict
import datetime

# Define local paths
DATA_PATH = "/Users/dianaterraza/Desktop/Data/"  # Update with your actual local data path
MODEL_PATH = "/Users/dianaterraza/Desktop/Data/model/model.xgb"  # Update with your actual local model path

def main():
    st.title("Corporación Favorita Sales Forecasting")

    # Load data and model
    df_stores, df_items, df_transactions, df_oil, df_holidays, df_train = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)

    # UI components for inputs
    store_id = st.selectbox("Store", df_stores['store_nbr'].unique())  # Dynamic store selection
    item_id = st.selectbox("Item", df_items['item_nbr'].unique())  # Dynamic item selection

    default_date = datetime.date(2014, 3, 1)  # Default to March 1, 2014
    min_date = datetime.date(2013, 1, 1)
    max_date = datetime.date(2014, 6, 7)
    date = st.date_input("Forecast Date", value=default_date, min_value=min_date, max_value=max_date)

    # Run prediction when button is clicked
    if st.button("Get Forecast"):
        input_data = preprocess_input_data(store_id, item_id, date, df_stores, df_items, df_train)
        prediction = predict(model, input_data)
        st.write(f"Predicted Sales for {date}: {prediction[0]}")

if __name__ == "__main__":
    main()
