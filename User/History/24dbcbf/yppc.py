from gc import collect

collect()
import time
import auth
from os import remove, rename
import _thread

collect()
from lib_mrequests import get

collect()
from hashlib import sha256

collect()


def get_sha256(file):
    """
    Returns the sha256 hash of the specified file.

    Args:
        file (str): The file to get the sha256 hash of.

    Returns:
        str: The sha256 hash of the specified file.
    """

    sha256_hash = sha256()
    with open(file, "rb") as f:
        while True:
            byte_block = f.read(1024)
            if not byte_block:
                break
            sha256_hash.update(byte_block)
    hash_digest = sha256_hash.digest()
    hex_digest = "".join("{:02x}".format(byte) for byte in hash_digest)
    return hex_digest


class WebServer:
    def __init__(self) -> None:
        """
        Initializes webserver.
        """
        self.database_downloaded = False

    def update_database_thread(self, url: str, headers: dict[str, str]) -> None:
        """
        Thread used to download the database asynchronously.

        Args:
            url (str): The url of the database.
            headers (dict[str, str]): The headers to send with the request.
        """

        # Send an HTTP GET request to retrieve the JSON file
        response = get(url, headers=headers)

        with auth.file_update_lock:
            # Check if the server has a newer version of the JSON file
            if response.status_code == 200:

                response.save("temp_records_file.json")

                # check if file exists
                try:
                    remove("records_file.json")
                except OSError:
                    pass
                rename("temp_records_file.json", "records_file.json")

                print("Database updated successfully.")
                # print contents of file
                print("[WEB] File contents:")
                with open("records_file.json", "r") as f:
                    print(f.read())
            elif response.status_code == 304:
                print("[WEB] Local database is up to date.")
            else:
                print("[WEB] Error updating database:", response.status_code)

        response.close()
        self.database_downloaded = True
        collect()

    def update_database(self, url: str, mac_address: str) -> bool:
        """
        Updates the database by downloading it from the specified url.

        Args:
            url (str): The url of the database.
            mac_address (str): The MAC address of the device.

        Returns:
            bool: True if the database was updated successfully, False otherwise.
        """

        if self.database_downloaded:
            print("[WEB] Database already downloaded.")
            return True

        block_size = 1024  # Adjust the block size as per your requirements

        # Prepare headers with the MAC address and If-None-Match
        headers = {
            # "If-None-Match": bytes(local_hash),
            "X-MAC-Address": mac_address,
        }

        try:
            local_hash = get_sha256("records_file.json")  # TODO add check in case file doesn't exist or is malformed
            print("[WEB] Local hash:", local_hash)
            headers["If-None-Match"] = local_hash
        except Exception as e:
            print("[WEB] Couldn't get file hash: " + e)

        _thread.start_new_thread(self.update_database_thread, (url, headers))

        start_time = time.time()
        while not self.database_downloaded and (time.time() - start_time) < 30:
            time.sleep(1)

        if not self.database_downloaded:
            print("[WEB] Database update timed out.")
            # send error to mqtt server
            return False
        return True


# Ram print check
# from micropython import mem_info

# print("[WEBSERVER] Ram check")
# print(mem_info())
