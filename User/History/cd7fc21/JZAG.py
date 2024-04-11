# Import mavutil
from pymavlink import mavutil
from drone_helpers import do_command, do_command_func

# Create the connection
master = mavutil.mavlink_connection("tcp:127.0.0.1:5762")
# Wait a heartbeat before sending commands
master.wait_heartbeat()

do_command(master, master.arducopter_arm, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)

# wait until arming confirmed (can manually check with master.motors_armed())
print("Waiting for the vehicle to arm")
master.motors_armed_wait()
print("Armed!")

# change mode
do_command_func(master, master.set_mode, 11)
