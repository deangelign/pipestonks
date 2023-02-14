# -*- coding: utf-8 -*-

import json
from typing import List
import warnings
from ..utils import *


def read_json_to_list(file_path: str) -> List[str]:
    """Read the given JSON file into a list.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A list containing the data from the given JSON file.
    """
    with open(file_path, "r") as f:
        content = json.load(f)

    list_of_stocks = content["stocks"]
    return list_of_stocks


def filter_dict_of_stocks_by_keys(
    dict_of_stocks: Dict[str, pd.DataFrame], keys: List[str]
) -> Dict[str, pd.DataFrame]:
    """Filter the given dictionary of stocks by the given keys.

    Args:
        dict_of_stocks: The dictionary of stocks to be filtered.
        keys: The keys to be used to filter the dictionary of stocks.

    Returns:
        The filtered dictionary of stocks.
    """
    filtered_dict = {}
    for key in keys:
        if key in dict_of_stocks:
            filtered_dict[key] = dict_of_stocks[key]
        else:
            warnings.warn(f"Key {key} not found in the dictionary of stocks.")

    return filtered_dict


if __name__ == "__main__":
    init_firebase_connection(get_secrets())
    root_folder = ""
    data_folder = os.path.join(root_folder, "br_stock_exchange/")
    output_file_format = get_storage_file_format()

    bucket = storage.bucket()  # storage bucket

    stocks_to_filter = read_json_to_list(".config/config.json")
    dict_of_stocks = {}
    list_objects = bucket.list_blobs(prefix=data_folder)
    for obj in list_objects:
        if obj.name.endswith("/"):
            continue
        if not obj:
            warnings.warn(f"File {obj.name} does not exists.")
            continue

        full_stock_df = load_dataframe_from_blob(obj)
        dict_of_stocks[obj.name] = full_stock_df
    filtered_stocks = filter_dict_of_stocks_by_keys(dict_of_stocks, stocks_to_filter)
