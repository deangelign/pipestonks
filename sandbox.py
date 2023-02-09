# flake8: noqa
# type: ignore

import firebase_admin
from firebase_admin import credentials, storage

import json
import os


# TODO create secrets for each argument.
config_dict = {
    "type": "service_account",
    "project_id": "pipestonks",
    "private_key_id": os.environ["private_key_id"],
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCmrTXqJzGc4ZRX\nr2JbonTBmxIHJWIq0I3xjDghcVW39nnftK9wRBWfn9BncxZo2gFrspbZuUtLpoqC\nFXsg2HKtegyFqChZOCLrlkua8BWGHXq5mgmEyz395UI3OyoivjCiuUGa/1VcLYx3\nMUY1cg6E5bD9UGi0eMbiNYNBloogoXULBt7xIvRiKPV1mtfgf3KWTb5Lhga1HHL6\n9tDeTbThU74EYu8yFpbi2gGf8osbDolp/bYihylpXgDTylGJO/Sd18aGbZRRHhoW\nUkJKaj1Wj+wWGLGnZovxC87BkWq0u1EUSXOe8arSVdUjOz7w2WtXoaVcStAhRx0p\nvP7uZjZpAgMBAAECggEAFoDVweSjR+dT6CfLe4w6P8ZwP2s+bwHj3CweivNai6EZ\nMMkcE6f5lGjXEhSBfswKO5Br2f7wqcjVM/1y94sJpp7PEH+U3QDdhBmxYPyPp2EV\nB/iCVv26w6v/jeXsyS6M8fdZonPdQjn7GD+cl0wsJii9xeGklVMmJPXtH/6TbRav\npaC7hgLSVD4X7jtXIB+U3kG7IZTRV+vpSjrym38bznSkacTU9aKZ3EioTjM6dItx\nZAmLx3neBavOLUyUeMB1g61j3jUYJ/rwh29t/1cPQsTmEXYFeb4RhkE+bf80OosT\nt1T1x64YFxXcVCYlJYVFFrTbQxCRYbjygjP3AcRjlQKBgQDSKKmlTojzlMwExApB\nIDOCIPZHe344UME7GwPWVxSkGvaGPY7Z39BHyxl85JB40nd+Su45q5KU0vk8v/br\nt3BvkJFkWmWcaiG004bIhMidi561qFP0Loi+GuOHGTGcziyTIkM1dVofofrAZv7e\nYEYXbbTUWGW0HJEWcdF9T0d5NQKBgQDLCHxlMWEtczg0p0HPd3mwTkQIN5ovGw5n\neLY/S2u/AM8OQu0lgLHZsF6HjpZ0vySYsduB5z6ai+G4ucnQNVankkPQfsOZHjUn\nNkhmYwVx4dZnglvEuCH/Nu9uDRVn2W2ObxukaYznPSyYt4GU9qpazrzPvpXbtxrr\nvfIj5V7i5QKBgQC7IM/lzKcvVPfQ6opC8RxMK3N/tNtv46AbM/PXv9Q2R3fpkhiH\nsb1wn8zDI3Xsz2LtBmVW3on3kF+zEy8XNlCcVnrPg5pkizAWJh8mnu1PMwoPsKGI\nlILX23NrUSiW+hJAtMppaGPmNMHk3hDlC459wAa66TcuCB28gX5KePQoXQKBgFk3\nS2I0Bp2wKZyJepOtzl04pxBtTmUwoG27T2oUvC3cLx/3LLn0CTK9G2y5rUdzzqMC\nVJCKiqnimbdQfGvvZDqSPRZa/ZxE99pZMgs7q+LU+B/X2ndg6h95hlB3k1Zk5o1W\nKXFfqDjcWsJosJpCoazxd2paNwOrPDsm5kX7vmw5AoGBALnrkcITmo12WeDQnhvx\nJ3kUG5YkStBIlFpMI28qUpxQjyLMC5upWBBmB9jhaf0fFU/bSXSu0m9ZjRuIepdT\nQJH63PQPpbo4L7b13jBEBjbGZfYOACedIAEuxVcZ2dNraWovKhPwp5dN+OR+oikz\nLa6R/i3r5hkZHLu4vdj5lUfT\n-----END PRIVATE KEY-----\n",
    "client_email": os.environ["client_email"],
    "client_id": os.environ["client_id"],
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.environ["client_x509_cert_url"],
}

print(config_dict)

print(f"Test")
print(config_dict["private_key"])
print(os.environ["private_key"])
print(config_dict["private_key"] == os.environ["private_key"])

cred = credentials.Certificate(config_dict)
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
