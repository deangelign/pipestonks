# -*- coding: utf-8 -*-
"""Utility functions for the project."""

import io
import os  # noqa
import firebase_admin  # type: ignore
from typing import Dict, Tuple
from firebase_admin import credentials, storage
import google  # type: ignore
import pandas as pd


def get_storage_file_format():
    """Returns the default format used to store the pandas.dataframe.

    Returns:
        The default format used to store the pandas.dataframe.
    """
    return ".parquet.gzip"


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

    Args:
        secrets : Dict[str, str]
            Dictionary containing the Firebase credentials.
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
    bucket: storage.bucket.Bucket,
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
