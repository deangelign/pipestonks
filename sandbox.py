import firebase_admin
from firebase_admin import credentials, storage

import sys
import json

# TODO create secrets for each argument.
config_dict = {
    "type": "service_account",
    "project_id": "pipestonks",
    "private_key_id": sys.argv[1],
    "private_key": sys.argv[2],
    "client_email": sys.argv[3],
    "client_id": sys.argv[4],
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": sys.argv[5],
}

config_file = ".config/config.json"
with (config_file, "w") as file:
    json.dump(config_dict, file)

cred = credentials.Certificate(config_file)
# default_app = firebase_admin.initialize_app(cred)
firebase_admin.initialize_app(
    cred, {"storageBucket": "pipestonks.appspot.com"}
)  # connecting to firebase

remote_file_path = "joao_test.txt"
local_file_path = "test.txt"
with open(local_file_path, "w") as file:
    file.write("This is a test!")

bucket = storage.bucket()  # storage bucket

blob = bucket.blob(remote_file_path)
blob.upload_from_filename(local_file_path)

new_local_file_path = "new" + local_file_path
blob.download_to_filename(new_local_file_path)
with open(new_local_file_path, "r") as file:
    str_file = file.readlines()
    print(str_file)
