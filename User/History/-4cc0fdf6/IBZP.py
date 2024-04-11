from helpers import close_windows_with_title_starting_with, check_if_window_exists_with_title_starting_with
import subprocess
import os

os.system("dronekit-sitl copter --home=41.934654,12.454560,17,353")