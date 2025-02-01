import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Define your local data path
DATA_PATH = "/Users/dianaterraza/Desktop/Data/"

def load_data(data_path=DATA_PATH):
    """
    Loads CSV files into DataFrames from the local file system.
    """

    # Define the paths for all the required data files
    files = {
        "stores": f"{data_path}stores.csv",  # Path for stores data
        "items": f"{data_path}items.csv",  # Path for items data
        "transactions": f"{data_path}transactions.csv",  # Path for transactions data
        "oil": f"{data_path}oil.csv",  # Path for oil prices data
        "holidays_events": f"{data_path}holidays_events.csv",  # Path for holidays and events data
        "train": f"{data_path}train.csv"  # Path for training data 
    }

    # Load each CSV file into a pandas DataFrame
    df_stores = pd.read_csv(files["stores"])
    df_items = pd.read_csv(files["items"])
    df_transactions = pd.read_csv(files["transactions"])
    df_oil = pd.read_csv(files["oil"])
    df_holidays = pd.read_csv(files["holidays_events"])

    # Filter and process the training data
    store_ids = df_stores[df_stores['state'] == 'Pichincha']['store_nbr'].unique()
    item_ids = [564533, 838216, 582865, 364606]
    max_date = '2014-04-01'

    # Read the CSV file in chunks for memory efficiency
    filtered_chunks = []
    chunk_size = 10 ** 6  # Adjust based on your system's memory capacity
    for chunk in pd.read_csv(files["train"], chunksize=chunk_size):
        chunk_filtered = chunk[
            (chunk['store_nbr'].isin(store_ids)) & 
            (chunk['item_nbr'].isin(item_ids)) & 
            (chunk['date'] < max_date)
        ]
        filtered_chunks.append(chunk_filtered)
        del chunk  # Free memory

    # Combine filtered chunks into a single DataFrame
    df_filtered = pd.concat(filtered_chunks, ignore_index=True)
    df_filtered = df_filtered.groupby(['store_nbr', 'item_nbr', 'date']).sum()['unit_sales'].reset_index()

    return df_stores, df_items, df_transactions, df_oil, df_holidays, df_filtered

def preprocess_input_data(store_id, item_id, split_date, df_stores, df_items, df_filtered):
    """
    Preprocesses input data into a format suitable for model prediction.
    """
    # Ensure date columns are in datetime format
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    split_date = pd.to_datetime(split_date)

    # Filter data for the selected store, item, and date
    df_filtered = df_filtered[
        (df_filtered['store_nbr'] == store_id) & 
        (df_filtered['item_nbr'] == item_id) & 
        (df_filtered['date'] >= split_date)
    ]

    # Create a complete date range
    full_date_range = pd.date_range(df_filtered['date'].min(), df_filtered['date'].max(), freq='D')
    df_filled = pd.DataFrame(index=full_date_range)
    df_filled.index.name = 'date'

    # Reindex and fill missing sales with 0
    df_filled = df_filtered.set_index('date').reindex(df_filled.index, fill_value=0).reset_index()

    # Add store and item information
    df_filled['store_nbr'] = store_id
    df_filled['item_nbr'] = item_id

    # Feature engineering
    df_filled['month'] = df_filled['date'].dt.month
    df_filled['day'] = df_filled['date'].dt.day
    df_filled['weekofyear'] = df_filled['date'].dt.isocalendar().week
    df_filled['dayofweek'] = df_filled['date'].dt.dayofweek
    df_filled['rolling_mean'] = df_filled['unit_sales'].rolling(window=7).mean()
    df_filled['rolling_std'] = df_filled['unit_sales'].rolling(window=7).std()
    df_filled['lag_1'] = df_filled['unit_sales'].shift(1)
    df_filled['lag_7'] = df_filled['unit_sales'].shift(7)
    df_filled['lag_30'] = df_filled['unit_sales'].shift(30)

    # Drop NaN rows
    df_filled.dropna(inplace=True)

    # Merge with metadata
    df_filled = df_filled.merge(df_stores, on='store_nbr', how='left').merge(df_items, on='item_nbr', how='left')

    # Encode categorical columns
    for col in ['city', 'state', 'type', 'family', 'class']:
        le = LabelEncoder()
        df_filled[col] = le.fit_transform(df_filled[col])

    return df_filled
