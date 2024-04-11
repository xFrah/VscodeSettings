from gc import collect
from microdot_asyncio import Microdot, Response, Request

collect()
import uasyncio as asyncio
import time
import auth
from os import remove, rename
import _thread

collect()
from mrequests import get

collect()
from hashlib import sha256

collect()

app: Microdot = Microdot()
Request.max_content_length = 1024 * 1024  # 1MB (change as needed)
print("[WEBSERVER] Started Microdot singleton")


def get_sha256(file):
    """
    Returns the sha256 hash of the specified file.

    Args:
        file (str): The file to get the sha256 hash of.

    Returns:
        str: The sha256 hash of the specified file.
    """

    # TODO make it asynchronous

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

    @app.route("/")
    async def get_info(req):
        """
        Sends the visible data as a json object.
        """
        # name = None
        # if req.method == "POST":
        #     name = req.form.get("name")
        return "No data available", 204, {"Content-Type": "text/plain"}
        # return visible_data + {"age": time.time() - push_timestamp}, 200, {"Content-Type": "application/json"}

    # @app.post("/database_update")
    async def database_update_upload(request):
        # print(request.headers)
        # obtain the filename and size from request headers
        filename = request.headers["Content-Disposition"].split("filename=")[1].strip('"')
        size = int(request.headers["Content-Length"])

        # sanitize the filename
        filename = filename.replace("/", "_")

        if filename == "records_file.json":
            with auth.file_update_lock:
                # write the file to the files directory in 1K chunks
                with open("temp_records_file.json", "wb") as f:
                    while size > 0:
                        chunk = await request.stream.read(min(size, 1024))
                        f.write(chunk)
                        size -= len(chunk)

                # check if file exists
                try:
                    remove("records_file.json")
                except OSError:
                    pass
                rename("temp_records_file.json", "records_file.json")

            print("Successfully saved file: " + filename)
            # print contents of file
            print("File contents:")
            with open("records_file.json", "r") as f:
                print(f.read())
            return "Local database updated", 200, {"Content-Type": "text/plain"}
        return "Invalid file", 400, {"Content-Type": "text/plain"}

    def update_database_thread(self, url: str, headers: dict[str, str]):
        """
        Thread used to download the database asynchronously.

        Args:
            url (str): The url of the database.
            headers (dict): The headers to send with the request.
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
                print("File contents:")
                with open("records_file.json", "r") as f:
                    print(f.read())
            elif response.status_code == 304:
                print("Local database is up to date.")
            else:
                print("Error updating database:", response.status_code)

        response.close()
        self.database_downloaded = True
        collect()

    def update_database(self, url, mac_address):

        # TODO make this asynchronous

        if self.database_downloaded:
            return

        block_size = 1024  # Adjust the block size as per your requirements

        # local_hash = get_sha256("records_file.json")  # TODO add check in case file doesn't exist or is malformed
        # print("Local hash:", local_hash)

        # Prepare headers with the MAC address and If-None-Match
        headers = {
            # "If-None-Match": bytes(local_hash),
            "X-MAC-Address": mac_address,
        }

        _thread.start_new_thread(self.update_database_thread, (url, headers))

    async def start(self) -> None:
        """
        Coroutine that starts the webserver.
        """
        asyncio.create_task(app.start_server(debug=True))


# Ram print check
# from micropython import mem_info

# print("[WEBSERVER] Ram check")
# print(mem_info())
