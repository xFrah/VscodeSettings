import utime
import os
from helpers import remove, rename, get_btree_sha256, get_configuration, get_last_update_info
import btree
from lib_mrequests import get
from sec import security_object as sec_obj


def check_for_updates(url: str, update_hours: list[int], mac_address: str):
    # file last_check.txt contains the timestamp of last check
    # file update_hours.txt contains the hour of the day when the check should be performed

    config = get_configuration()
    last_update_info = get_last_update_info()
    local_time_int = last_update_info["last_check"] if last_update_info else 0

    # TODO check if the wake reason is hard reset
    current_hour = utime.localtime()[3]

    hours = [current_hour] if not update_hours else update_hours

    # if current hour in hours and last check was more than an hour ago
    if current_hour in hours and utime.mktime(utime.localtime()) - local_time_int > 3600:
        print("[UPDATER] Updating...")
        sha256 = get_btree_sha256()
        response = get(
            url,
            headers={
                "If-None-Match": sha256,
                "X-MAC-Address": mac_address,
            },
        )

        with sec_obj.file_update_lock:
            if response.status_code == 200:
                response.save("temp.txt")
                with open("badge_hashes", "w+b") as f:
                    db = btree.open(f)
                    with open("temp.txt", "r") as file:
                        # get line by line and put in btree
                        i = 0
                        for line in file:  # TODO does this contain the newline character?
                            l_ = line.strip()
                            db[l_] = None
                            i += 1
                            print(f"[WEB] Added badge {l_} to database.")
                    db.close()
                remove("temp.txt")

            elif response.status_code == 304:
                print("[WEB] Local database is up to date.")
            else:
                print("[WEB] Error updating database:", response.status_code)

        response.close()
    elif current_hour not in hours:
        next_hour = min([i for i in hours if i > local_time[3]])
        print(f"[UPDATER] Next check for updates will be at {next_hour:0>2}:00.")
    else:
        print("[UPDATER] Last check was less than an hour ago.")
