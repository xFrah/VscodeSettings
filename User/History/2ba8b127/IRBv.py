import datetime
import sys
import time
import traceback
from typing import Any, Dict, List, Tuple, Union
from flask import Flask, jsonify, request, abort, make_response
from flask_jwt_extended import JWTManager, get_jwt, create_access_token, verify_jwt_in_request
import threading
from argon2.exceptions import VerifyMismatchError
import database as db
from my_exceptions import (
    NoKitException,
    BadgeNeededException,
    CompanyNotFoundException,
    UserNotFoundException,
    UnresponsiveKitException,
    RegistrationTimeoutException,
)

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "your-secret-key"  # Change this!
jwt = JWTManager(app)
app.secret_key = "super_secret_key"  # Used to encrypt the session

kit_columns = db.get_product_types()
ROLE_KIT, ROLE_ADMIN, ROLE_USER = "012"


@app.errorhandler(Exception)
def handle_exception(e):
    status_code = 500
    if isinstance(e, NoKitException) or isinstance(e, CompanyNotFoundException):
        status_code = 404
    elif isinstance(e, VerifyMismatchError) or isinstance(e, UserNotFoundException):
        status_code = 401
    elif isinstance(e, BadgeNeededException) or isinstance(e, UnresponsiveKitException) or isinstance(e, RegistrationTimeoutException):
        status_code = 412
    elif isinstance(e, ValueError) or isinstance(e, db.psycopg2.IntegrityError):
        status_code = 400
    _, _, exc_traceback = sys.exc_info()
    traceback_details = traceback.extract_tb(exc_traceback)
    # Get the last traceback item (most recent call last)
    last_traceback = traceback_details[-1]
    filename = last_traceback.filename
    line_number = last_traceback.lineno
    print(f"{e.__class__.__name__}({filename}.{line_number}): {e}")
    return jsonify(error=str(e)), status_code


def requires_clearance(required_role):
    """Check if user has the required role."""
    clearance = get_jwt()["clearance_level"]  # Get the entire decoded JWT token
    if type(required_role) == list:
        if clearance not in required_role:
            print(f"Role {clearance} is not allowed. Requireds: {required_role}")
            abort(403)
        return
    if clearance != required_role:
        print(f"Role {clearance} is not allowed. Required: {required_role}")
        abort(403)  # Forbidden


def requires_arguments(*args) -> Union[str, List[str]]:
    """Check if all the arguments are present in the request."""
    args_list = []
    for arg in args:
        request_arg = request.args.get(arg)
        if not request_arg:
            print(f"Missing {arg} parameter.")
            abort(400, description=f"{arg} parameter is required.")
        args_list.append(request_arg)
    return args_list


def requires_form_entries(*args) -> Union[str, List[str]]:
    """Check if all the arguments are present in the request."""
    args_list = []
    for arg in args:
        request_arg = request.form.get(arg)
        if not request_arg:
            raise ValueError(f"Missing {arg} parameter.")
        args_list.append(request_arg)
    return args_list


@app.route("/access", methods=["GET", "POST"])
def access_route():
    """Route for authentication."""
    print("Form:", request.form)
    form = request.form
    if not form or ("username" not in form or "password" not in form):
        raise ValueError("Missing credentials.")

    username, password = form["username"], form["password"]
    if not username or not password:
        raise ValueError("Missing credentials.")

    user = db.get_user(username)

    username, hash_password, clearance, company_id = user
    db.hash_login(username, hash_password, password)

    access_token = create_access_token(identity=username, additional_claims={"company_id": company_id, "clearance_level": str(clearance)})
    return jsonify(access_token=access_token)


@app.route("/get_devices", methods=["GET"])
def get_devices():
    """
    A client sends a GET request to this route to get the devices of a company.
    """
    verify_jwt_in_request()  # Check if the JWT is valid
    company_id = get_jwt()["company_id"]

    # list of tuples (mac_address, alias)
    devices: List[Tuple[str, str]] = db.get_kits_for_company(company_id)
    dict_devices = {mac_address: alias for mac_address, alias in devices}
    return dict_devices


@app.route("/get_kit", methods=["POST"])
def get_kit_info():
    """
    A client sends a GET request to this route to get the info of a single kit.
    """
    verify_jwt_in_request()
    requires_clearance([ROLE_USER, ROLE_ADMIN])  # admin or kit role

    start = time.time()

    [kit_mac_address] = requires_form_entries("kit_mac_address")

    products = db.get_medical_kit_content(kit_mac_address)

    # item_type: n_occurrences
    occurrences: Dict[str, int] = db.product_type_constraints()

    # ids of the products that are missing
    missing = {
        iid for iid, date in db.get_disappearing_items(kit_mac_address).items() if (datetime.datetime.now() - date).total_seconds() > 15
    }

    for p in products:
        if p.product_id not in missing:
            occurrences[p.product_type] -= 1

    # item_type: number of missing items
    missing_types = {item_type: difference for item_type, difference in occurrences.items() if difference > 0}
    contents = {p.product_id: (p.expiration_date, p.product_id in missing, p.product_type) for p in products}

    kit_info_dict: Dict[str, Any] = {
        "kit_mac_address": kit_mac_address,
        "contents": contents,
        "missing_types": missing_types,
        "is_offline": kit_mac_address not in db.kit_reports,
    }
    print(kit_info_dict)

    print(f"get_kit_info took {time.time() - start} seconds.")

    return kit_info_dict


@app.route("/register_item", methods=["POST"])
def register_item():
    """
    A client sends a POST request to this route to register an item.
    The server then waits for the kit to send the unregistered items on the air.
    If they are found and meet the criteria, the closest one is registered.
    """
    verify_jwt_in_request()  # Check if the JWT is valid
    requires_clearance(ROLE_ADMIN)  # admin role

    [kit_mac_address, item_type, expiration_date] = requires_form_entries("kit_mac_address", "item_type", "expiration_date")

    if item_type not in kit_columns:
        raise ValueError("Invalid item_type.")

    print(f"Registering item {item_type} for kit {kit_mac_address}, expires in {expiration_date}.")

    possible_item_ids = db.get_unregistered_items(kit_mac_address, timeout=15)

    if len(possible_item_ids) > 1:
        # sort by rssi, highest first
        possible_item_ids.sort(key=lambda x: x[1], reverse=True)
        # check if first spans at least 15 dBm more than second
        if possible_item_ids[0][1] - possible_item_ids[1][1] < 7:
            print("Multiple unregistered items found for kit.")
            abort(412, description="Multiple unregistered items found for kit.")

    item_id = possible_item_ids[0][0]

    db.register_item(item_id, expiration_date, kit_mac_address, item_type)

    response = make_response(f"Item {item_id} of type {item_type} registered successfully in kit {kit_mac_address}.")
    response.status_code = 200
    return response


@app.route("/register_kit_kitside", methods=["POST"])
def register_kit_kitside():
    """
    A kit sends a POST request to this route to register itself.

    """
    pass


@app.route("/register_kit_appside", methods=["POST"])
def register_kit_appside():
    """
    The app sends this post to receive an api token for the kit.
    """
    pass


@app.route("/getfile", methods=["GET"])
def serve_file():
    """
    This route is used to download the file containing the badges of a company.
    In this way the client doesn't need to make a request to the server every time it needs to authenticate a badge.
    """
    [mac_address, received_sha256] = requires_arguments("mac_address", "sha256")  # TODO check if its a valid sha256

    if not db.check_valid_mac(mac_address):
        abort(400, description="Invalid mac_address.")

    company = db.get_company_from_kit(mac_address)
    if company is None:
        abort(400, description="Kit not found.")

    company_id, company_sha256 = company
    if company_sha256 is None:
        print(f"Sha256 of company {company_id} is NULL.")

    badges = db.get_badges_for_company(company_id)
    if badges is None:
        abort(503, description="No badges found for company.")

    if received_sha256 != company_sha256:
        # TODO this goes all in the ram, maybe it's better to write to a file
        badge_ids_string = "\n".join(badge_id[0] for badge_id in badges)
        response = make_response(badge_ids_string)
        response.headers["Content-Disposition"] = "attachment; filename=badges.txt"
        return response

    response = make_response()
    response.status_code = 304  # Not Modified
    return response


def _run_flask(host, port):
    app.run(host=host, port=port)


def run_flask_thread(host="0.0.0.0", port=8080):
    threading.Thread(target=_run_flask, args=(host, port), daemon=True).start()
