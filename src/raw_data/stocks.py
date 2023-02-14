import os  # noqa
import firebase_admin  # type: ignore
import investpy  # type: ignore
import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore
import warnings
import google  # type: ignore
import io
import json
from typing import List, Dict, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from tqdm import tqdm  # type: ignore
from firebase_admin import credentials, storage


class DataGenerationType(Enum):
    """A class `DataGenerationType` is an `Enum` that enumerates the options for generating data.

    It has two values:

    * `ALL` (generates all available data) and;
    * `DATE_RANGE`  generates data within a specific date range.

    """

    ALL = 1
    DATE_RANGE = 2


def list_all_stock_symbols() -> List[str]:
    """Returns a list of all stock symbols.

    Returns:
        A list of all stock symbols.
    """
    list_symbols = investpy.stocks.get_stocks_list(country="Brazil")
    list_symbols.sort()
    list_symbols = [x + ".SA" for x in list_symbols]
    return list_symbols


def get_storage_file_format():
    """Returns the default format used to store the pandas.dataframe.

    Returns:
        The default format used to store the pandas.dataframe.
    """
    return ".parquet.gzip"


def get_range_dates(left_offset: int = 0, right_offset: int = 0) -> Tuple[str, str]:
    """Returns a tuple of datetime objects representing a range of dates.

    with optional offsets from the current date.
    Args:
        left_offset (int, optional): The number of days to subtract from the current date to get
            the starting date of the range. Defaults to 0.

        right_offset (int, optional): The number of days to add to the current date to get the
            ending date of the range. Defaults to 0.

    Returns:
        Tuple[str, str]: A tuple of two strings representing the start and end
            of the date range.
    """
    left = (datetime.now() + timedelta(days=left_offset)).strftime("%Y-%m-%d")
    right = (datetime.now() + timedelta(days=right_offset)).strftime("%Y-%m-%d")

    start_time = str(left)
    end_time = str(right)
    return (start_time, end_time)


def retrieve_stocks_data(
    stock_name: Union[str, List[str]],
    period: str = None,
    date_range: Tuple[str, str] = None,
):
    """Retrieves stock data for a given stock symbol or a list of symbols.

    Args:
        stock_name (Union[str, List[str]]): A stock symbol or a list of symbols for which data is
            to be retrieved.

        period (str, optional): The time period to retrieve data for. The available options
            are: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max. Defaults to None.

        date_range (Tuple[datetime, datetime], optional): A tuple of two strings
            specifying the start and end dates of the date range to retrieve data for.
            Defaults to None.

    Returns:
        pandas.DataFrame: A Dataframe containing the retrieved stock data.

    Raises:
        ValueError: If neither `period` nor `date_range` is set.
    """
    if period:
        data = yf.download(
            tickers=stock_name,
            period=period,
            interval="1d",
            ignore_tz=False,
            # group_by = 'ticker',
            auto_adjust=True,
            repair=False,
            prepost=True,
            threads=True,
            proxy=None,
        )
    elif date_range:
        data = yf.download(
            tickers=stock_name,
            start=date_range[0],
            end=date_range[1],
            interval="1d",
            ignore_tz=False,
            # group_by = 'ticker',
            auto_adjust=True,
            repair=False,
            prepost=True,
            threads=True,
            proxy=None,
        )
    else:
        raise ValueError(
            "Neither `period` nor `date_range` is set. Please set a value for one of them."
        )

    return data


def get_secrets() -> dict:
    """Retrieve the secrets for accessing the Firebase account.

    Returns:
        A dictionary of the secrets for accessing the Firebase account.
    """
    config_dict = {
        "type": os.environ["PUBLISHER_TYPE"],
        "project_id": os.environ["PUBLISHER_PROJECT_ID"],
        "private_key_id": os.environ["PUBLISHER_PRIVATE_KEY_ID"],
        "private_key": os.environ["PUBLISHER_PRIVATE_KEY"],
        "client_email": os.environ["PUBLISHER_CLIENT_EMAIL"],
        "client_id": os.environ["PUBLISHER_CLIENT_ID"],
        "auth_uri": os.environ["PUBLISHER_AUTH_URI"],
        "token_uri": os.environ["PUBLISHER_TOKEN_URI"],
        "auth_provider_x509_cert_url": os.environ["PUBLISHER_AUTH_PROVIDER_X509_CERT_URL"],
        "client_x509_cert_url": os.environ["PUBLISHER_CLIENT_X509_CERT_URL"],
    }
    return config_dict


def init_firebase_connection(secrets: Dict[str, str]) -> None:
    """Initialize connection to Firebase using the provided credentials in a dictionary.

    Parameters
    ----------
    secrets : Dict[str, str]
        Dictionary containing the Firebase credentials.

    Returns
    -------
    None
    """
    cred = credentials.Certificate(secrets)
    firebase_admin.initialize_app(
        cred, {"storageBucket": "pipestonks.appspot.com"}
    )  # connecting to firebase


def get_blob(list_objects: google.api_core.page_iterator.HTTPIterator, target_name: str):
    """Retrieve the blob with the specified target name from the list of objects.

    Args:
        list_objects: A page iterator of Google Cloud Storage objects.
        target_name: The target name of the desired blob.

    Returns:
        The blob with the specified target name if found, else None.
    """
    for obj in list_objects:
        if not obj.name.endswith("/"):
            if obj.name.endswith(target_name):
                return obj

    return None


def convert_parquet_bytes_2_dataframe(parquet_bytes: bytes):
    """Convert the given parquet bytes into a pandas DataFrame.

    Args:
        parquet_bytes: The parquet bytes to be converted.

    Returns:
        A pandas DataFrame created from the given parquet bytes.
    """
    pq_file = io.BytesIO(parquet_bytes)
    df = pd.read_parquet(pq_file)
    return df


def load_dataframe_from_filename(
    firebase_filename: str,
) -> Tuple[pd.DataFrame, google.cloud.storage.blob.Blob]:
    """Load a pandas DataFrame from the given Firebase filename.

    Args:
        firebase_filename: The filename of the desired data in Firebase.

    Returns:
        A tuple containing the loaded pandas DataFrame and the corresponding blob.
    """
    blob = bucket.blob(firebase_filename)
    existing_content = blob.download_as_bytes()
    existing_df = convert_parquet_bytes_2_dataframe(existing_content)
    return existing_df, blob


def load_dataframe_from_blob(blob: google.cloud.storage.blob.Blob) -> pd.DataFrame:
    """Load a dataframe from a Google Cloud Storage blob.

    Args:
        blob: The Google Cloud Storage blob.

    Returns:
        The loaded dataframe.
    """
    existing_content = blob.download_as_bytes()
    existing_df = convert_parquet_bytes_2_dataframe(existing_content)
    return existing_df


def save_dataframe(df: pd.DataFrame, blob: google.cloud.storage.blob.Blob) -> None:
    """Save a dataframe to a Google Cloud Storage blob.

    Args:
        df: The dataframe to be saved.
        blob: The Google Cloud Storage blob.
    """
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="auto", compression="gzip")
    blob.upload_from_string(buffer.getvalue())


if __name__ == "__main__":
    init_firebase_connection(get_secrets())
    root_folder = ""
    data_folder = os.path.join(root_folder, "br_stock_exchange/")
    output_file_format = get_storage_file_format()

    bucket = storage.bucket()  # storage bucket

    generation_type = DataGenerationType.DATE_RANGE
    stock_symbols = list_all_stock_symbols()
    dates = get_range_dates(-1, 1)

    for stock_symbol in tqdm(stock_symbols):
        stock_df = pd.DataFrame()
        basename = stock_symbol + output_file_format
        filename = os.path.join(data_folder, basename)

        if generation_type == DataGenerationType.ALL:
            stock_df = retrieve_stocks_data(stock_symbol, period="max")

        elif generation_type == DataGenerationType.DATE_RANGE:
            list_objects = bucket.list_blobs(prefix=data_folder)
            blob = get_blob(list_objects, filename)

            if not blob:
                warnings.warn(
                    f"File {filename} does not exists. It is not possible append the information in"
                    f" DataFrame. The stock {stock_symbol} will be ignored"
                )
                continue

            full_stock_df = load_dataframe_from_blob(blob)
            stock_df = retrieve_stocks_data(stock_symbol, date_range=dates)

            if not stock_df.empty:
                stock_df = pd.concat([full_stock_df, stock_df]).drop_duplicates()
            else:
                warnings.warn(f"Empty dataframe for {filename}.")
        else:
            raise TypeError(f"Invalid `DataGenerationType`: {generation_type}")

        if not stock_df.empty:
            save_dataframe(stock_df, blob)

    extra_info = {"last_update": datetime.now().strftime("%Y-%m%d %H:%M:%S")}
    filename = os.path.join(root_folder, "info.json")
    serialized_dictionary = json.dumps(extra_info)
    blob = storage.bucket().blob(filename)
    blob.upload_from_string(serialized_dictionary, content_type="application/json")
