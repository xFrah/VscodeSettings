# Import mavutil
from pymavlink import mavutil
from drone_helpers import do_command, do_command_func

# Create the connection
master = mavutil.mavlink_connection("tcp:127.0.0.1:5762")
# Wait a heartbeat before sending commands
master.wait_heartbeat()

# change mode
print("Changing mode...")
do_command_func(master, master.set_mode, ["GUIDED"], mavutil.mavlink.MAV_CMD_DO_SET_MODE)
# master.set_mode("GUIDED")

do_command_func(master, master.arducopter_arm, [], mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)

# wait until arming confirmed (can manually check with master.motors_armed())
print("Waiting for the vehicle to arm")
master.motors_armed_wait()
print("Armed!")

# takeoff
print("Taking off!")
do_command(master, [mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 20])

master.mav.mission_item_send(0, 0, 0, frame,
   mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 2, 0, 0,
   0, 0, 0, location.lat, location.lon,
   alt)