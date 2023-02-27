import os  # noqa
import investpy  # type: ignore
import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore
import warnings
import json
from typing import List, Dict, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from tqdm import tqdm  # type: ignore
from firebase_admin import storage  # type: ignore

from pipestonks.connection.firebase_util import (
    get_storage_file_format,
    get_secrets,
    init_firebase_connection,
    get_blob,
    load_dataframe_from_blob,
    save_dataframe,
)

from pipestonks.raw_data.publish_data_utils import (
    list_all_stock_symbols,
    get_range_dates,
    retrieve_stocks_data,
)


class DataGenerationType(Enum):
    """A class `DataGenerationType` is an `Enum` that enumerates the options for generating data.

    It has two values:

    * `ALL` (generates all available data) and;
    * `DATE_RANGE`  generates data within a specific date range.

    """

    ALL = 1
    DATE_RANGE = 2


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
