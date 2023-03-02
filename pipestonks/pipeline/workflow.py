import os
import pandas as pd  # type: ignore
import joblib  # type: ignore

from typing import List, Iterable, Dict, Tuple, Any
import json
from pipestonks.connection.firebase_util import (
    get_storage_file_format,
    get_secrets,
    init_firebase_connection,
    get_blob,
    load_dataframe_from_blob,
    get_temporary_folder,
)

from datetime import datetime, timedelta
from firebase_admin import storage  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression  # type: ignore
from sklearn.metrics import mean_absolute_error  # type: ignore
from tqdm import tqdm  # type: ignore


def get_target_stocks():
    """Returns a list of target stock symbols for filtering data.

    Returns:
    List of strings representing the target stock symbols.
    """
    # read_json_to_list(".config/config.json")
    return ["PETR4", "CPLE6", "ITSA4"]


def get_filtered_data(
    list_objects: Iterable, target_stocks_list: List[str]
) -> Dict[str, Tuple[str, pd.DataFrame]]:
    """Filters data from a list of objects based on the target stocks.

    Args:
    - list_objects (Iterable): An iterable object containing data to be filtered.
    - target_stocks_list (List[str]): A list of target stock symbols.

    Returns:
    A dictionary containing the filtered data for each target stock symbol.
    The dictionary keys represent the target stock symbol and the values are a tuple
    containing the object name and the filtered DataFrame.
    """
    filtred_info = {}
    for obj in list_objects:
        if obj.name.endswith("/"):
            continue

        for stock_symbol in target_stocks_list:
            if stock_symbol in obj.name:
                df = load_dataframe_from_blob(obj)
                filtred_info[stock_symbol] = (obj.name, df)
    return filtred_info


def remove_nan(df: pd.DataFrame):
    """Removes NaN values from a DataFrame.

    Args:
    - df (pd.DataFrame): A DataFrame containing NaN values.

    Returns:
    None.
    """
    df.interpolate(method="linear", inplace=True)
    df.fillna(method="ffill", inplace=True)


def compute_sma(
    df: pd.DataFrame, target_column: str = "Close", periods: List[int] = [5, 10, 20, 50, 100, 200]
):
    """Computes Simple Moving Averages (SMA) for a target column in a DataFrame.

    Args:
    - df (pd.DataFrame): A DataFrame containing the target column.
    - target_column (str): The name of the target column. Default is 'Close'.
    - periods (List[int]): A list of periods to compute the SMA. Default is [5,10,20,50,100,200].

    Returns:
    The input DataFrame with new columns added for each SMA period computed.
    """
    for sma_period in periods:
        indicator_name = f"SMA_{sma_period}"
        df[indicator_name] = df[target_column].rolling(sma_period).mean()
    return df


def compute_Bollinger_bands(
    df: pd.DataFrame,
    target_column: str = "Close",
    periods: List[int] = [20, 10],
    std_values: List[float] = [2, 1],
):
    """Computes Bollinger Bands for a target column in a DataFrame.

    Args:
    - df (pd.DataFrame): A DataFrame containing the target column.
    - target_column (str): The name of the target column. Default is 'Close'.
    - periods (List[int]): A list of periods to compute the Bollinger Bands. Default is [20, 10].
    - std_values (List[float]): A list of standard deviation values for the upper and lower bands.
        Default is [2, 1].

    Returns:
    The input DataFrame with new columns added for each Bollinger Band computed.
    """
    for period in periods:
        for std_value in std_values:
            df[f"BBand_Up_{period}_{std_value}"] = (
                df[target_column].rolling(period).mean()
                + std_value * df[target_column].rolling(period).std()
            )
            df[f"BBand_Down_{period}_{std_value}"] = (
                df[target_column].rolling(period).mean()
                - std_value * df[target_column].rolling(period).std()
            )
    return df


def compute_Donchian_channels(
    df: pd.DataFrame,
    high_column: str = "High",
    low_column: str = "Low",
    periods: List[int] = [5, 10, 20, 50, 100, 200],
):
    """Computes Donchian Channels for a high and low column in a DataFrame.

    Args:
    - df (pd.DataFrame): A DataFrame containing the high and low columns.
    - high_column (str): The name of the high column. Default is 'High'.
    - low_column (str): The name of the low column. Default is 'Low'.
    - periods (List[int]): A list of periods to compute the Donchian Channels.
        Default is [5,10,20,50,100,200].

    Returns:
    The input DataFrame with new columns added for each Donchian Channel computed.
    """
    for channel_period in periods:
        up_name = f"Donchian_Channel_Up_{channel_period}"
        down_name = f"Donchian_Channel_Down_{channel_period}"

        df[up_name] = df[high_column].rolling(channel_period).max()
        df[down_name] = df[low_column].rolling(channel_period).min()

    return df


def apply_lags(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
    """Shifts the 'Close' column of a DataFrame by a given number of lags.

    Generates new columns with the shifted data, drops missing values, and returns a
    new DataFrame with the shifted data and a target column.

    Args:
        df: A DataFrame with a 'Close' column.
        lags: A list of integers representing the number of time periods to shift the 'Close'
            column by. Default is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].

    Returns:
        A DataFrame with shifted 'Close' columns, as well as a 'target' column representing the
          value of the 'Close' column 5 time periods in the future.
    """
    newdata = df["Close"].to_frame()
    for lag in lags:
        shift = lag
        shifted = df.shift(shift)
        shifted.columns = [f"{column}_shifted_by_{shift}" for column in shifted.columns]
        newdata = pd.concat((newdata, shifted), axis=1)

    return newdata


def create_train_dataset(df: pd.DataFrame, forward_lag: int = 1, prediction_column: str = "target"):
    """Create a train dataset with lagged values of a given column as features.

    Args:
        df (pandas.DataFrame): The original dataset.

        forward_lag (int, optional): The number of periods to shift the target column forward.
            Defaults to 1.

        prediction_column (str, optional): The name of the target column. Defaults to "target".

    Returns:
        pandas.DataFrame: The train dataset with the target column shifted and NaN rows removed.
    """
    df[prediction_column] = df["Close"].shift(-forward_lag)
    df = df.dropna()

    return df


def extract_features(df):
    """Applies a series of technical analysis functions to a DataFrame.

    Args:
        df: A DataFrame containing financial market data.

    Returns:
        A new DataFrame with the results of technical analysis functions applied to the input
            DataFrame.
    """
    df = compute_sma(df)
    df = compute_Bollinger_bands(df)
    df = compute_Donchian_channels(df)
    df = apply_lags(df)
    df = df.dropna(axis=1, how="all")

    return df


def feature_selection(X: pd.DataFrame, y: pd.DataFrame, n_features: int) -> pd.DataFrame:
    """Performs feature selection on the input data.

    Args:
        X: A pandas DataFrame with the features to be selected.
        y: Prediction column.
        n_features: An integer representing the number of features to be selected.

    Returns:
        A pandas DataFrame with the n_features most representative.
    """
    selector = SelectKBest(f_regression, k=n_features)
    selector.fit(X, y)
    selected_features = selector.get_support(indices=True)
    return selected_features


def evaluate_models(X_train, y_train, X_test, y_test):
    """Evaluate different machine learning models using a grid search.

    return the ML model with the lowest mean absolute error (MAE).

    This method evaluates the following machine learning models:
    - Random Forest
    - Gradient Boosting Tree Regressor
    - K Nearest Neighbors Regressor
    - Neural Network

    For each model, a grid search is performed to find the best combination of hyperparameters
    using the mean absolute error as the scoring metric. The following hyperparameters are
    tuned for each model:
    - Random Forest: Number of estimators (10, 50, 100).
    - Gradient Boosting Tree Regressor: Number of estimators (10, 50, 100) and
        learning rate (0.1, 0.5, 1.0).
    - K Nearest Neighbors Regressor: Number of neighbors (3, 5, 7).
    - Neural Network: Size of the hidden layers ((50,), (100,), (50, 50)).

    The best model found by the grid search is returned. This method requires the Sci-kit Learn
    library to be installed.

    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training input samples.
    y_train : array-like of shape (n_samples,)
        Target values for training input samples.
    X_test : array-like of shape (n_samples, n_features)
        Test input samples.
    y_test : array-like of shape (n_samples,)
        Target values for test input samples.

    Returns:
    --------
    best_model : estimator
        Best machine learning model found by the grid search.
    """
    models = {
        "Random Forest": RandomForestRegressor(),
        # "Gradient Boosting": GradientBoostingRegressor(),
        "KNN": KNeighborsRegressor(),
        # "Neural Network": MLPRegressor(),
    }

    params = {
        "Random Forest": {"n_estimators": [10, 50, 100]},
        # "Gradient Boosting": {"n_estimators": [10, 50, 100], "learning_rate": [0.1, 0.5, 1.0]},
        "KNN": {"n_neighbors": [3, 5, 7]},
        # "Neural Network": {"hidden_layer_sizes": [(50,), (100,), (50, 50)]},
    }

    best_model = None
    best_mae = float("inf")
    for name, model in models.items():
        grid_search = GridSearchCV(model, params[name], cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"{name} MAE: {mae:.2f}")
        if mae < best_mae:
            best_mae = mae
            best_model = grid_search.best_estimator_

    return {
        "model": best_model,
        "MAE": best_mae,
    }


def get_data_2_infer(features: pd.DataFrame, selected_feats_name: List[str]):
    """Extracts the selected features of the most recent row.

    Args:
        features (pd.DataFrame): Input features DataFrame.
        selected_feats_name (List[str]): List with the names of the selected features.

    Returns:
        pd.DataFrame: DataFrame with the most recent row and the selected features for inference.
    """
    data_infer = df_feats.tail(1)[selected_feats_name]
    return data_infer


def generate_future_dates(df: pd.DataFrame, periods: int) -> pd.Series:
    """This function takes a DataFrame with a DatetimeIndex and discovers its frequency.

    It then generates future dates based on the discovered frequency and the number of
    periods specified.

    Args:
        df (pd.DataFrame): A DataFrame with a DatetimeIndex
        periods (int): The number of future periods to generate

    Returns:
        pd.Series: A pandas Series with the future dates
    """
    # Get the index of the DataFrame and convert it to a pandas DatetimeIndex
    idx = pd.DatetimeIndex(df.index)

    # Compute the difference between consecutive dates in the index and find the mode
    freq = pd.Series(idx[1:] - idx[:-1]).mode()[0]

    # Generate future dates based on the discovered frequency and the number of periods specified
    future_dates = pd.date_range(start=idx[-1] + freq, periods=periods, freq=freq)

    return future_dates


if __name__ == "__main__":
    init_firebase_connection(get_secrets())

    root_folder = ""
    data_folder = os.path.join(root_folder, "br_stock_exchange/")
    models_folder = os.path.join(root_folder, "ML_models/")
    reports_folder = os.path.join(root_folder, "reports/")
    temp_folder = get_temporary_folder()

    output_file_format = get_storage_file_format()
    bucket = storage.bucket()  # storage bucket

    list_objects = bucket.list_blobs(prefix=data_folder)
    stocks_to_filter = get_target_stocks()

    filtred_info = get_filtered_data(list_objects, stocks_to_filter)
    summary = {}
    for key, value in tqdm(filtred_info.items()):
        print(key)
        df_all_data = value[1]
        df = df_all_data

        df_feats = extract_features(df)
        df_dataset = create_train_dataset(df_feats, forward_lag=1)  # predict tomorrow

        X = df_dataset.drop("target", axis=1)
        y = df_dataset["target"]

        selected_feats_indices = feature_selection(X, y, 50)
        selected_feats_name = X.columns[selected_feats_indices]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = X_train[selected_feats_name]
        X_test = X_test[selected_feats_name]

        # Evalute models and pick the best one
        res_summary = evaluate_models(X_train, y_train, X_test, y_test)

        # forecast
        input_data = get_data_2_infer(df_feats, selected_feats_name)
        future_dates_pred = generate_future_dates(df_feats, 1)
        prediction = res_summary["model"].predict(input_data)
        diff = prediction - input_data["Close"]

        # creating report
        res_summary["predction"] = {}
        res_summary["predction"]["date"] = list(future_dates_pred)
        res_summary["predction"]["value"] = list(prediction)
        res_summary["predction"]["diff"] = list(diff)

        # store the results localy
        summary[key] = res_summary

    print("Update models")
    for key in tqdm(summary.keys()):
        temp_file = os.path.join(temp_folder, f"{key}.joblib")
        filename_model_cloud = os.path.join(models_folder, key, f"{key}.joblib")
        filename_info_cloud = os.path.join(models_folder, key, "model_info.json")

        joblib.dump(summary[key]["model"], temp_file)

        with open(temp_file, "rb") as f:
            model_bytes = f.read()

        blob = bucket.blob(filename_model_cloud)
        blob.upload_from_string(model_bytes, content_type="application/octet-stream")

        extra_info = {
            "training_date": datetime.now().strftime("%Y_%m_%d %H:%M:%S"),
            "model_type": str(summary[key]["model"]),
            "MAE": summary[key]["MAE"],
        }
        serialized_dictionary = json.dumps(extra_info)
        blob = storage.bucket().blob(filename_info_cloud)
        blob.upload_from_string(serialized_dictionary, content_type="application/json")

    print("Creating report")
    report: Dict[str, Any] = {}
    for key in tqdm(summary.keys()):
        report[key] = {}
        report[key]["prediciton"] = str(summary[key]["predction"])

        summary[key]["predction"]
    report[key]["published_date"] = str(datetime.now().strftime("%Y_%m_%d %H:%M:%S"))

    today_date = str(datetime.now().strftime("%Y_%m_%d"))
    filename_report = os.path.join(reports_folder, f"report_{today_date}.json")
    serialized_dictionary = json.dumps(report)
    blob = storage.bucket().blob(filename_report)
    blob.upload_from_string(serialized_dictionary, content_type="application/json")
