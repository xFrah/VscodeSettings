

if not check_if_window_exists_with_title_starting_with("Copter"):
    command = r"title Copter && dronekit-sitl copter --home=41.934654,12.454560,17,353"
    proc = subprocess.Popen(f'start cmd /k "{command}"', shell=True)
if not check_if_window_exists_with_title_starting_with("Mavproxy"):
    command = r"title Mavproxy && mavproxy.py --master tcp:127.0.0.1:5760 --out udp:127.0.0.1:1450 --out udp:127.0.0.1:1440"
    proc = subprocess.Popen(f'start cmd /k "{command}"', shell=True)
