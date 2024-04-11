# Import mavutil
from pymavlink import mavutil
from drone_helpers import do_command

# Create the connection
master = mavutil.mavlink_connection('tcp:127.0.0.1:5762')
# Wait a heartbeat before sending commands
master.wait_heartbeat()

do_command(master, master.arducopter_arm, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM)

# wait until arming confirmed (can manually check with master.motors_armed())
print("Waiting for the vehicle to arm")
master.motors_armed_wait()
print('Armed!')

do_command(master, [mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
   0, 0, 0, 0, 0, 0, 0, 10], mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)