from contextlib import closing
from datetime import datetime
import hashlib
import os
import sys
import psycopg2
import threading
import time
from argon2 import PasswordHasher
from typing import Any, Dict, List, Set, Tuple, Union
from my_exceptions import (
    BadgeNeededException,
    CompanyNotFoundException,
    NoKitException,
    UnresponsiveKitException,
    UserNotFoundException,
    RegistrationTimeoutException,
)


def get_db_credentials() -> Tuple[str, str]:
    try:
        db_username = os.environ["DB_USERNAME"]
        db_password = os.environ["DB_PASSWORD"]
    except KeyError as e:
        sys.exit(f"Error: Environment variable {e} not set.")
    else:
        return db_username, db_password


DB_NAME = "data"
DB_USERNAME, DB_PASSWORD = get_db_credentials()

reports_lock = threading.Lock()
file_lock = threading.Lock()
hasher = PasswordHasher()

# TODO the fact that this is in the ram, means that we can't use multiple instances of the server
kit_reports: Dict[str, "KitReport"] = {}  # key is mac address
SESSION_START_TIMESTAMP = datetime.now()


class KitReport:
    def __init__(self, json_data: Dict[str, Any]):
        self.unregistered_items: Dict[str, int] = {}  # list is (item_id, rssi)
        self.missing_items: Set[str] = set()  # str is item type
        self.null_items: Set[str] = set()  # str is item type
        self.missing_items_dict: Dict[str, str] = {}  # key is item_id, value is item_type
        self.timestamp = time.time()
        self.mac = json_data["mac"]
        self.closed = json_data["status"] == "closed"
        self.authed = json_data["authenticated"]
        self.valid = None
        try:
            self.decode(json_data)
            self.valid = True
        except Exception as e:
            print(f"Error while decoding kit report: {e}")
            self.valid = False
        print(f"Kit report decoded in {time.time() - self.timestamp:.04f} seconds.\n")

    def is_valid(self) -> int:
        # 0 = not yet decoded, 1 = valid, 2 = invalid
        return 0 if self.valid is None else 1 if self.valid else 2

    def validate_json_rssi(self, rfid_rssi: Dict[str, int]):
        # check if all ids have length 24
        for rfid, rssi in rfid_rssi.items():
            if len(rfid) != 24:
                raise ValueError(f"Invalid RFID {rfid}, wrong length {len(rfid)} != 24.")
            # check that rssi is integer
            if type(rssi) != int:
                raise ValueError(f"Invalid RSSI {rssi}, not an integer.")

    def decode(self, json_data: Dict[str, Any]):
        print(f"Decoding kit report for {self.mac}, {json_data['status']}...")
        products = get_medical_kit_content(self.mac)

        rfid_rssi: Dict[str, int] = dict(json_data["rfid"])  # json_data["rfid"] is a list of (rfid, rssi) tuples
        self.validate_json_rssi(rfid_rssi)

        all_item_ids = {product.product_id for product in products}

        self.unregistered_items = {rfid: rssi for rfid, rssi in rfid_rssi.items() if rfid not in all_item_ids}
        print(f"I see {len(self.unregistered_items)} unregistered items: {', '.join(self.unregistered_items.keys())}")

        if self.authed:
            return

        disappearing: Dict[str, datetime] = get_disappearing_items(self.mac)

        # dictionary of item_type: number of items
        occurrences: Dict[str, int] = {product.product_type: 0 for product in products}
        for product in products:
            occurrences[product.product_type] += 1

        removed: Set[str] = set()

        with closing(connect_db()) as conn:
            cursor = conn.cursor()

            for iid, date in disappearing.items():
                # we don't wanna risk killing items because the server was down for a while
                if date < SESSION_START_TIMESTAMP:
                    cursor.execute("DELETE FROM temp_auto_removed WHERE item_id = %s", (iid,))
                    print(f"[INFO] Item {iid} removed from temp_auto_removed because it was older than session start time.")
                elif iid in all_item_ids and iid in rfid_rssi:
                    cursor.execute("DELETE FROM temp_auto_removed WHERE item_id = %s", (iid,))
                    print(f"[INFO] Item {iid} removed from temp_auto_removed because it reappeared.")
                elif iid not in all_item_ids:
                    cursor.execute("DELETE FROM temp_auto_removed WHERE item_id = %s", (iid,))
                    print(f"[INFO] Item {iid} removed from temp_auto_removed because it was not in the kit.")
                else:
                    # if date is older than 30 seconds, kill it
                    if (datetime.now() - date).total_seconds() > 30:
                        if occurrences[product.product_type] > 1:
                            occurrences[product.product_type] -= 1
                            removed.add(iid)
                            cursor.execute("DELETE FROM temp_auto_removed WHERE item_id = %s", (iid,))
                            cursor.execute("DELETE FROM product WHERE product_id = %s", (iid,))
                            print(f"[INFO] Auto-removed item {iid} of type {product.product_type}.")

            for product in products:
                if product.product_id in removed:
                    continue
                if product.product_id not in rfid_rssi:
                    cursor.execute("INSERT OR IGNORE INTO temp_auto_removed (mac, item_id) VALUES (%s, %s)", (self.mac, product.product_id))
            conn.commit()


def get_unregistered_items(kit_mac_address: str, timeout=15.0) -> List[Tuple[str, int]]:
    """
    Waits for the next mqtt message containing the unregistered items for the specified kit, then returns it.
    """
    flag = False
    start = time.time()
    while time.time() - start < timeout:
        with reports_lock:
            global kit_reports
            kit_report = kit_reports.get(kit_mac_address, None)
            if kit_report and kit_report.timestamp > start and kit_report.is_valid() == 1:
                flag = True
                if not kit_report.authed:
                    raise BadgeNeededException()
                res = list(kit_report.unregistered_items.items())
                if res:
                    print(f"Unregistered items for {kit_mac_address} found, kit is open, report is valid.")
                    return res
                print(f"No unregistered items found for {kit_mac_address}.")
        time.sleep(1)
    if not flag:
        raise UnresponsiveKitException()
    raise RegistrationTimeoutException()


def get_disappearing_items(kit_mac_address: str) -> Dict[str, datetime]:
    # select where mac in table temp_auto_removed in sqlite
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT item_id, last_seen FROM temp_auto_removed WHERE mac = %s", (kit_mac_address,))

    disappearing_items = cursor.fetchall()

    close_db(conn)

    if disappearing_items:
        print(f"Disappearing items: {disappearing_items}")
        return {item_id: datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for item_id, timestamp in disappearing_items}

    return {}


def clear_old_reports():
    """
    Clears the reports that are older than 15 seconds.
    """
    now = time.time()
    with reports_lock:
        global kit_reports
        leng = len(kit_reports)
        # TODO maybe do in-place
        kit_reports = {mac: report for mac, report in kit_reports.items() if report.timestamp > now - 60}
    if leng != len(kit_reports):
        print(f"Removed {leng - len(kit_reports)} old reports.")


def push_kit_report(json_data):
    """
    Pushes a kit report to the kit report dictionary.
    """
    with reports_lock:
        global kit_reports
        new = KitReport(json_data)
        kit_reports[new.mac] = new


def clearkit(mac: str):
    """
    Make all the attributes of the kit with the specified MAC address NULL.
    """
    conn = connect_db()
    cursor = conn.cursor()

    # now its all in products table
    cursor.execute("DELETE FROM products WHERE mac = %s", (mac,))

    close_db(conn)


def hash_login(user: str, hash_password: str, password: str):
    """
    Hashes the password for login.
    """

    hasher.verify(hash_password, password)
    if hasher.check_needs_rehash(hash_password):
        update_hash_password(user, password)


def update_hash_password(username: str, password: str):
    """
    Updates the hash of the password.
    """
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('UPDATE "user" SET password = %s WHERE username = %s', (hasher.hash(password), username))

    close_db(conn)


def get_product_types() -> List[str]:
    """
    Returns the names of the columns of the kit contents table.
    """
    conn = connect_db()
    cursor = conn.cursor()

    # select all type_names from product_type table
    cursor.execute("SELECT type_name FROM product_type")

    product_types = cursor.fetchall()

    close_db(conn)

    return [product_type[0] for product_type in product_types]


def change_kit_alias(mac: str, alias: str):
    """
    Changes the alias of the specified kit.
    """
    check_valid_mac(mac)

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("UPDATE medical_kit SET alias = %s WHERE mac = %s", (alias, mac))

    rowcount = cursor.rowcount + 0

    close_db(conn)

    if not rowcount:
        raise NoKitException(mac)


def product_type_constraints():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT type_name, number FROM product_type")

    constraints = cursor.fetchall()

    close_db(conn)

    return {product_type: min_n for product_type, min_n in constraints}


def setup_database():
    """
    Sets up the database with the necessary tables.
    """

    if os.path.isfile(DB_NAME):
        print(f"Database {DB_NAME} already exists.")
        return

    conn = connect_db()
    cursor = conn.cursor()

    with open("schema.sql") as f:
        cursor.execute(f.read())

    close_db(conn)
    print(f"Database {DB_NAME} is set up with the necessary tables.")


def connect_db():
    return psycopg2.connect(
        host="localhost",
        database=DB_NAME,
        user=DB_USERNAME,
        password=DB_PASSWORD,
    )


def close_db(conn):
    conn.commit()
    conn.close()


def update_users_sha256(company_id: int):
    """
    Updates the SHA256 hash of the users of the specified company. The hash is computed on the badge IDs.
    """

    badge_ids = get_badges_for_company(company_id)

    if not badge_ids:
        print(f"No badges found for company {company_id}.")
        return

    try:
        badge_ids_string = "\n".join(badge_id[0] for badge_id in badge_ids)
        sha256 = compute_sha256_str(badge_ids_string)

    except Exception as e:
        print(f"Error while computing SHA256: {e}")
        sha256 = None

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("UPDATE companies SET users_sha256 = %s WHERE index_id = %s", (sha256, company_id))

    rowcount = cursor.rowcount + 0

    close_db(conn)

    if not rowcount:
        raise CompanyNotFoundException(company_id)

    if sha256 is None:
        print(f"SHA256 of users of company {company_id} set to NULL.")
    else:
        print(f"SHA256 of users of company {company_id} updated to {sha256}.")


def get_badges_for_company(company_id: int) -> List[str]:
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('SELECT badge_id FROM "user" WHERE company_id = %s AND badge_id IS NOT NULL', (company_id,))
    badges = cursor.fetchall()

    close_db(conn)
    return badges  # type: ignore


def get_kit(mac: str, company_id: int) -> List[str]:
    """
    Returns the kit from the database.
    """
    check_valid_mac(mac)

    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM medical_kit WHERE mac = %s AND company_id = %s", (mac, company_id))
    kit = cursor.fetchone()
    close_db(conn)
    if not kit:
        raise NoKitException(mac)
    return kit  # type: ignore


def check_valid_mac(mac: str):
    """
    Checks if the MAC address is valid. Raises a ValueError if it's not.
    """
    if len(mac) != 17:
        print(f"Invalid MAC address length {len(mac)}.")
        raise ValueError(f"Invalid MAC address length {len(mac)}.")
    for i in range(2, 17, 3):
        if mac[i] != ":":
            raise ValueError(f"Invalid MAC address, no colon at position {i}.")


def compute_sha256(file_path: str) -> str:
    """
    Computes and returns the SHA256 hash of a file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_sha256_str(string: str) -> str:
    """
    Computes the SHA256 hash of a string.
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(string.encode())
    return sha256_hash.hexdigest()


def add_user(username: str, password: str, badge_id: Union[str, None], clearance: str, company_id: int):
    """
    Adds a user to the database.
    Raises IntegrityError if the username already exists or company doesn't exist.
    """
    if username.isdigit():
        raise ValueError(f"Name must not be a number.")
    if not clearance.isdigit():
        raise ValueError(f"Clearance must be a number or NULL.")
    try:
        company_id = int(company_id)
    except ValueError:
        raise ValueError(f"Company ID must be a number.")

    conn = connect_db()
    cursor = conn.cursor()

    if badge_id == "NULL":
        badge_id = None

    hashed_password = hasher.hash(password)

    cursor.execute(
        'INSERT INTO "user" (username, password, badge_id, clearance, company_id) VALUES (%s, %s, %s, %s, %s)',
        (username, hashed_password, badge_id, clearance, company_id),
    )

    close_db(conn)

    update_users_sha256(company_id)  # TODO maybe make this a trigger


def remove_user(username: str, company_id: int):
    """
    Removes a user from the database.
    The company_id is needed to update the SHA256 hash of the users of the company.
    """

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('DELETE FROM "user" WHERE username = %s', (username,))

    rowcount = cursor.rowcount + 0
    close_db(conn)

    if not rowcount:
        raise UserNotFoundException(username)

    update_users_sha256(company_id)  # TODO maybe make this a trigger


def get_user(username: str) -> Tuple[str, str, str, str]:
    """
    Returns a tuple containing the username, password, clearance, and company ID.
    """
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('SELECT username, password, clearance, company_id FROM "user" WHERE username = %s', (username,))
    user = cursor.fetchone()

    close_db(conn)

    if not user:
        raise UserNotFoundException(username)

    return user  # type: ignore


def get_users() -> List[Tuple[str, str, str, str]]:
    """
    Returns a list of tuples containing the username, password, clearance and company ID.
    """
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute('SELECT username, password, clearance, company_id FROM "user"')
    users = cursor.fetchall()

    close_db(conn)

    return users  # type: ignore


def get_kits_for_company(company_id: int) -> List[Tuple[str, str]]:
    """
    Returns a list of tuples containing the MAC address and the company ID.
    """
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT mac, alias FROM medical_kit WHERE company_id = %s", (company_id,))
    kits = cursor.fetchall()

    close_db(conn)

    return kits  # type: ignore


def add_company(company_name: str):
    """
    Adds a company to the database.
    """
    if company_name.isdigit():
        print(f"Name must not be a number.")
        return

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO companies (company_name) VALUES (%s)", (company_name,))

    close_db(conn)


def remove_company(company_id: int):
    """
    Removes a company from the database.
    """
    try:
        company_id = int(company_id)
    except ValueError:
        raise ValueError(f"Company ID must be a number.")

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM companies WHERE index_id = %s", (company_id,))

    rowcount = cursor.rowcount + 0

    close_db(conn)

    if not rowcount:
        raise CompanyNotFoundException(company_id)


def get_companies() -> List[tuple]:
    """
    Returns a list of tuples containing the companies details.
    """
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM companies")
    companies = cursor.fetchall()

    close_db(conn)

    return companies


def register_item(item_id: str, expiry_date: str, kit_mac_address: str, item_type: str):
    """
    Puts an item in the contents of the specified kit, in the column specified by item_type.
    """
    check_valid_mac(kit_mac_address)

    try:
        datetime.strptime(expiry_date, "%d-%m-%Y")
    except ValueError:
        raise ValueError(f"Invalid date format {expiry_date}.")

    if item_type not in kit_content_columns:
        raise ValueError(f"Unknown item type {item_type}.")

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO product (product_id, product_type, expiration_date, mac) VALUES (%s, %s, %s, %s)",
        (item_id, item_type, expiry_date, kit_mac_address),
    )

    rowcount = cursor.rowcount + 0

    close_db(conn)

    if not rowcount:
        raise NoKitException(kit_mac_address)


def add_medical_kit(mac: str, company_id: int):
    """
    Adds a medical kit to the database.
    """
    check_valid_mac(mac)

    try:
        company_id = int(company_id)
    except ValueError:
        raise ValueError(f"Company ID must be a number.")

    with closing(connect_db()) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO medical_kit (
                mac, 
                company_id
            ) 
            VALUES (%s, %s)
            """,
            (mac, company_id),
        )
        conn.commit()


def remove_medical_kit(mac: str, company_id: int):
    """
    Removes a medical kit from the database.
    """
    check_valid_mac(mac)

    try:
        company_id = int(company_id)
    except ValueError:
        raise ValueError(f"Company ID must be a number.")

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM medical_kit WHERE mac = %s AND company_id = %s", (mac, company_id))

    rowcount = cursor.rowcount + 0

    close_db(conn)

    if not rowcount:
        raise NoKitException(mac)


def get_medical_kits() -> List[Tuple[str, str, str]]:
    """
    Returns a list of tuples containing the MAC address and the company ID.
    """
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT mac, company_id, alias FROM medical_kit")
    medical_kits = cursor.fetchall()

    close_db(conn)

    return medical_kits  # type: ignore


class Product:
    def __init__(self, product_id: str, mac: str, product_type: str, expiration_date: str):
        self.mac = mac
        self.product_id = product_id
        self.product_type = product_type
        self.expiration_date = expiration_date

    def __repr__(self):
        return f"Product({self.mac}, {self.product_id}, {self.product_type}, {self.expiration_date})"


def get_medical_kit_content(mac: str) -> List[Product]:
    """
    Returns a tuple containing the content of the specified medical kit.
    """
    check_valid_mac(mac)

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT product_id, mac, product_type, expiration_date FROM product WHERE mac = %s", (mac,))

    kit_content = cursor.fetchall()

    close_db(conn)

    if not kit_content:
        raise NoKitException(mac)

    return [Product(*content) for content in kit_content]


if __name__ == "__main__":
    setup_database()

kit_content_columns = get_product_types()
