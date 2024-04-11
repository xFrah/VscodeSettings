# Import mavutil
import time
from pymavlink import mavutil
from drone_helpers import do_command, do_command_func, get_location, is_armed

# Create the connection
master = mavutil.mavlink_connection("tcp:127.0.0.1:5762")
# Wait a heartbeat before sending commands
master.wait_heartbeat()

master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_POSITION,
    1,
    1,
)


# change mode
print("Changing mode...")
do_command_func(
    master, master.set_mode, ["GUIDED"], mavutil.mavlink.MAV_CMD_DO_SET_MODE
)
# master.set_mode("GUIDED")

do_command_func(
    master, master.arducopter_arm, [], mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
)

# wait until arming confirmed (can manually check with master.motors_armed())
print("Waiting for the vehicle to arm")
master.motors_armed_wait()
print("Armed!")

# takeoff
# print("Taking off!")
target_altitude = 10

try:
    do_command(
        master,
        [mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, target_altitude],
    )
except Exception as e:
    print("Takeoff failed.")

while is_armed(master):
    lat, lon, alt = get_location(master)
    print(" Altitude: ", alt)
    if alt >= target_altitude * 0.95:  # Trigger just below target alt.
        print("Reached target altitude")
        break
    time.sleep(1)

# send mission item NAV_LAND
master.mav.mission_item_send(
    master.target_system,
    master.target_component,
    0,  # seq
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
    mavutil.mavlink.MAV_CMD_NAV_LAND,
    2,  # current - guided-mode request
    0,  # autocontinue
    0,  # p1
    0,  # p2
    0,  # p3
    0,  # p4
    lat,  # latitude
    lon,  # longitude
    alt,  # altitude
    mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
)

# mav.mission_item_int_send(
# target_system,
# target_component,
# 0, # seq
# mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
# mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
# 2, # current - guided-mode request
# 0, # autocontinue
# 0, # p1
# 0, # p2
# 0, # p3
# 0, # p4
# int(loc.lat *1e7), # latitude
# int(loc.lng *1e7), # longitude
# desired_relative_alt, # altitude
# mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
