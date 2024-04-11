import time
from typing import List, Tuple
from terminal import CMD
from mqtt import mqtt_client
import json
import database as db
import webserver

_CLOSED_REPORT = "00"
_OPEN_REPORT = "01"
opmodes = [_CLOSED_REPORT, _OPEN_REPORT]
kit_columns = db.get_kit_contents_columns()


def decode_mqtt_message(msg):
    # protocol
    # opmode = chars 0 to 1
    # mac = chars 2 to 18
    # data = chars 19 to end

    opmode = msg[0:2]
    mac = msg[2:19]
    data = msg[19:]

    if opmode not in opmodes:
        print("Unknown opmode.")
    if not db.check_if_valid_mac(mac):
        print("Invalid MAC address.")

    if opmode == _CLOSED_REPORT:
        try:
            print(data)
            json_data = json.loads(data)
            if json_data["mac"] != mac or not db.check_if_valid_mac(mac):
                print("Invalid MAC address.")
                return
            db.push_kit_report(mac, json_data)

        except Exception as e:
            print(f"Invalid JSON: {e}")
            return None


def main():
    mqtt = mqtt_client()
    webserver.run_flask_thread()
    time.sleep(1)
    cmd = CMD()

    while True:
        if cmd.data_ready():
            command, args = cmd.get_command()
            if command == "company":
                if args[0] == "add":
                    db.add_company(args[1])
                elif args[0] == "remove":
                    db.remove_company(args[1])

            elif command == "listcompanies":
                companies = db.get_companies()
                if len(companies) != 0:
                    print("Companies:")
                    for company in db.get_companies():
                        print(f"- {company[0]}: {company[1]}")
                else:
                    print("No companies found.")

            elif command == "kit":
                if args[0] == "add":
                    db.add_medical_kit(args[1], args[2])
                elif args[0] == "remove":
                    db.remove_medical_kit(args[1], args[2])
                else:
                    print(f"Unknown action {args[0]}.")

            elif command == "listkits":
                kits = db.get_medical_kits()
                if len(kits) != 0:
                    print("Kits:")
                    for kit in kits:
                        print(f"- {kit[0]}{'(' + str(kit[2]) + ')' if kit[2] else ''}: {kit[1]}")
                else:
                    print("No kits found.")

            elif command == "adduser":
                db.add_user(args[0], args[1], args[2], args[3], args[4])

            elif command == "removeuser":
                db.remove_user(args[0], args[1])

            elif command == "ping":
                print("Pong")

            elif command == "changealias":
                db.change_kit_alias(args[0], args[1])

            elif command == "registeritem":
                db.register_item(args[1], args[0], args[2])

            elif command == "listuser":
                users = db.get_users()
                if len(users) != 0:
                    print("Users:")
                    for user in users:
                        print(f"- {user[0]}: {user[1]}")
                else:
                    print("No users found.")

        if mqtt.data_ready():
            msg = mqtt.get_message()
            decode_mqtt_message(msg)

        db.clear_old_reports()  # TODO maybe this is being done too often
        time.sleep(0.1)


if __name__ == "__main__":
    main()
