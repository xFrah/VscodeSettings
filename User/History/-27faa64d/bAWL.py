import json
import micropydatabase
import os

file_update_lock = asyncio.Lock()


def auth(idd):
    # open json file, get the dict and check if id is in key

    auth_dict = json.load(open("auth.json", "r"))
    print(auth_dict)
    if idd in auth_dict.keys():
        return True
    return False


def get_sha256(file):
    """
    Returns the sha256 hash of the specified file.

    Args:
        file (str): The file to get the sha256 hash of.

    Returns:
        str: The sha256 hash of the specified file.
    """
    import hashlib

    sha256_hash = hashlib.sha256()
    with open(file, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
