import utime
import os
from helpers import remove, rename, get_btree_sha256
import btree
from lib_mrequests import get
from sec import security_object as sec_obj


def check_for_updates():
    # file last_check.txt contains the timestamp of last check
    # file update_hours.txt contains the hour of the day when the check should be performed

    # TODO check if the wake reason is hard reset

    # check if the file exists
    if not os.path.isfile("last_check.txt"):
        # if not, create it
        with open("last_check.txt", "w") as f:
            v = str(utime.mktime(utime.localtime()))
            print("[UPDATER] Creating last_check.txt with value:", v)
            f.write(v)

    try:
        with open("last_check.txt", "r") as f:
            local_time_int = int(f.read())
        local_time = utime.localtime(local_time_int)
        print("[UPDATER] Last check was on:", local_time)
    except Exception as e:
        print("[UPDATER] Error reading last_check.txt:", e)
        remove("last_check.txt")

    hours = [local_time[3]]
    try:
        with open("update_hours.txt", "r") as f:
            hours = f.read().split(",")
            print("[UPDATER] Update hours:", hours)
    except Exception as e:
        print("[UPDATER] Error reading update_hours.txt:", e)
        remove("update_hours.txt")

    # if current hour in hours and last check was more than an hour ago
    if local_time[3] in hours and utime.mktime(utime.localtime()) - local_time_int > 3600:
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
            # Check if the server has a newer version of the JSON file
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

        print(local_time)
    elif local_time[3] not in hours:
        next_hour = min([i for i in hours if i > local_time[3]])
        print(f"[UPDATER] Next check for updates will be in {next_hour:0>2}:00.")
    else:
        print("[UPDATER] Last check was less than an hour ago.")
