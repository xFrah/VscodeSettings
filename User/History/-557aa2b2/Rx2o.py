import threading
from typing import Any, Callable, Dict, List, Tuple
from pymavlink import mavutil
import math
import time


class MissionError(Exception):
    def __init__(self, msg):
        self.msg = msg


class CommandException(Exception):
    def __init__(self, msg):
        self.msg = msg


class FetchException(Exception):
    def __init__(self, msg):
        self.msg = msg


class Drone:
    def __init__(
        self,
        connection_string: str,
        starting_mode: str = "STABILIZE",
        poll_frequency: float = 0.1,
        start: bool = True,
    ):
        self.packets = dict()
        self.poll_frequency = poll_frequency
        self.master = mavutil.mavlink_connection(connection_string)
        self.master.wait_heartbeat()
        self.inverse_mode_mapping = {v: k for k, v in self.master.mode_mapping().items()}
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
            1,
            1,
        )
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
            1,
            1,
        )

        threading.Thread(target=self.event_handler, daemon=True).start()
        self.packets: Dict[str, Tuple[Any, float]] = {
            "HEARTBEAT": (None, time.time()),
            "GLOBAL_POSITION_INT": (None, time.time()),
        }
        self.wait_for_setup()

        if start:
            self.change_mode("RTL")
            while True:
                if not self.is_armed:
                    print("Disarmed!")
                    break
                _, _, alt = self.location
                print(f" Altitude: {alt:.02f}m")
                time.sleep(1)
            self.change_mode(starting_mode)

    def wait_for_setup(self, timeout: int = 20):
        # no element in packets is None
        start = time.time()
        while True:
            if all([x[0] is not None for x in self.packets.values()]):
                print("Setup complete")
                return
            if time.time() - start > timeout:
                raise FetchException("Setup failed")
            time.sleep(1)

    def event_handler(self):
        while True:
            msg = self.master.recv_match(blocking=True)
            if msg is None:
                continue

            if msg.get_type() == "HEARTBEAT" and msg.to_dict()["type"] != 2:
                continue
            # if msg.get_type() == "SYS_STATUS":
            #     # get pre arm check status
            #     self.print_full_status(msg.onboard_control_sensors_health)
            self.packets[msg.get_type()] = msg, time.time()

    def print_full_status(self, bitmap):
        field_names = {
            1: "MAV_SYS_STATUS_SENSOR_3D_GYRO",
            2: "MAV_SYS_STATUS_SENSOR_3D_ACCEL",
            4: "MAV_SYS_STATUS_SENSOR_3D_MAG",
            8: "MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE",
            16: "MAV_SYS_STATUS_SENSOR_DIFFERENTIAL_PRESSURE",
            32: "MAV_SYS_STATUS_SENSOR_GPS",
            64: "MAV_SYS_STATUS_SENSOR_OPTICAL_FLOW",
            128: "MAV_SYS_STATUS_SENSOR_VISION_POSITION",
            256: "MAV_SYS_STATUS_SENSOR_LASER_POSITION",
            512: "MAV_SYS_STATUS_SENSOR_EXTERNAL_GROUND_TRUTH",
            1024: "MAV_SYS_STATUS_SENSOR_ANGULAR_RATE_CONTROL",
            2048: "MAV_SYS_STATUS_SENSOR_ATTITUDE_STABILIZATION",
            4096: "MAV_SYS_STATUS_SENSOR_YAW_POSITION",
            8192: "MAV_SYS_STATUS_SENSOR_Z_ALTITUDE_CONTROL",
            16384: "MAV_SYS_STATUS_SENSOR_XY_POSITION_CONTROL",
            32768: "MAV_SYS_STATUS_SENSOR_MOTOR_OUTPUTS",
            65536: "MAV_SYS_STATUS_SENSOR_RC_RECEIVER",
            131072: "MAV_SYS_STATUS_SENSOR_3D_GYRO2",
            262144: "MAV_SYS_STATUS_SENSOR_3D_ACCEL2",
            524288: "MAV_SYS_STATUS_SENSOR_3D_MAG2",
            1048576: "MAV_SYS_STATUS_GEOFENCE",
            2097152: "MAV_SYS_STATUS_AHRS",
            4194304: "MAV_SYS_STATUS_TERRAIN",
            8388608: "MAV_SYS_STATUS_REVERSE_MOTOR",
            16777216: "MAV_SYS_STATUS_LOGGING",
            33554432: "MAV_SYS_STATUS_SENSOR_BATTERY",
            67108864: "MAV_SYS_STATUS_SENSOR_PROXIMITY",
            134217728: "MAV_SYS_STATUS_SENSOR_SATCOM",
            268435456: "MAV_SYS_STATUS_PREARM_CHECK",
            536870912: "MAV_SYS_STATUS_OBSTACLE_AVOIDANCE",
            1073741824: "MAV_SYS_STATUS_SENSOR_PROPULSION",
            2147483648: "MAV_SYS_STATUS_EXTENSION_USED",
        }

        # Iterate through each bit and print the corresponding field name if the bit is set
        for key in field_names:
            if not (bitmap & key):
                print(field_names[key])

    def get_packet(
        self,
        packet_type: str,
        no_older_than: float = 1,
        timeout: int = 15,
        pop: bool = False,
        raise_exception: bool = True,
    ):
        start = time.time()
        while True:
            if (msg := self.packets.get(packet_type, None)) and msg[0]:
                # if packet arrived later than no_older_than seconds ago
                if time.time() - self.packets[packet_type][1] < no_older_than:
                    # print(f"Got {packet_type} packet")
                    if pop:
                        return self.packets.pop(packet_type)[0]
                    return self.packets[packet_type][0]
                # else:
                #     print(f"Packet {packet_type} is too old, {time.time() - self.packets[packet_type][1]:.02f}s")
            if time.time() - start > timeout:
                if raise_exception:
                    raise FetchException(f"Failed to get {packet_type} packet")
                else:
                    return None
            time.sleep(0.05)

    @property
    def is_armed(self) -> bool:
        msg = self.get_packet("HEARTBEAT", no_older_than=0.5)
        return msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

    @property
    def pre_arm_check(self) -> bool:
        msg = self.get_packet("SYS_STATUS", no_older_than=0.5)
        return msg.onboard_control_sensors_health & 268435456

    @property
    def mode(self) -> str:
        msg = self.get_packet("HEARTBEAT", no_older_than=0.5)
        return self.inverse_mode_mapping.get(msg.custom_mode, "Unknown")

    @property
    def location(self) -> Tuple[float, float, float]:
        msg = self.get_packet("GLOBAL_POSITION_INT", no_older_than=0.5)
        return (msg.lat * 1e-7, msg.lon * 1e-7, msg.relative_alt * 1e-3)

    def distance_to_waypoint(self, lat: float, lon: float):
        """
        Gets distance in metres to the current waypoint.
        It returns None for the first waypoint (Home location).
        """
        lat2, lon2, _ = self.location
        distancetopoint = get_distance_metres((lat2, lon2), (lat, lon))
        return distancetopoint

    def change_mode(self, mode: str, timeout: int = 10):
        do_command_func(
            self,
            self.master.set_mode,
            [mode],
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            failed_rebounce=0.25,
            timeout=1,
            retries=10,
        )
        start = time.time()
        while True:
            if self.mode == mode:
                print(f"Mode changed to {mode}")
                return True
            if time.time() - start > timeout:
                raise CommandException(f"Mode change to {mode} failed")
            print(f"Waiting for mode {mode}, currently in {self.mode}")
            time.sleep(1)

    def arm(self, timeout: int = 10):
        if not self.pre_arm_check:
            raise CommandException("Pre arm check failed")
        print("Waiting for arming...")
        do_command_func(
            self,
            self.master.arducopter_arm,
            [],
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            timeout=1.5,
            retries=20,
            failed_rebounce=0.25,
        )
        start = time.time()
        while not self.is_armed:
            if time.time() - start > timeout:
                raise CommandException("Arming failed")
            time.sleep(1)
        print("Armed")

    def arm_and_takeoff(self, target_altitude, timeout: int = 60):
        """
        Arms vehicle and fly to aTargetAltitude.
        """

        self.change_mode("GUIDED")

        print("Arming motors")
        self.arm()

        self.takeoff(target_altitude)

        start = time.time()
        while self.is_armed:
            _, _, alt = self.location
            print(f" Altitude: {alt:.02f}m")
            if alt >= target_altitude * 0.95:  # Trigger just below target alt.
                print("Reached target altitude")
                break
            if time.time() - start > timeout:
                return False
            time.sleep(1)
        return True

    def takeoff(self, target_altitude):
        do_command(
            self,
            [mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, target_altitude],
        )

    def close(self):
        self.master.close()

    def __del__(self):
        self.close()


def get_distance_metres(p1, p2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the
    earth's poles. It comes from the ArduPilot test code:
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    lat1, lon1 = p1
    lat2, lon2 = p2
    dlat = lat2 - lat1
    dlong = lon2 - lon1
    return math.sqrt((dlat * dlat) + (dlong * dlong)) * 1.113195e5


def progress_debug(drone: Drone, waypoint):
    alt = drone.location[2]
    if alt:
        if alt < waypoint.alt_target * 0.95 or alt > waypoint.alt_target * 1.05:
            print(f"Altitude: {alt:.02f}m")
        else:
            print(f"Distance to waypoint: {drone.distance_to_waypoint(waypoint.lat, waypoint.lon):.02f}m")


def do_command(drone: Drone, cmd: List[int], timeout: int = 10, retries: int = 5, failed_rebounce: float = 5):
    """
    Sends a command via pymavlink then waits for a COMMAND_ACK. If failed it retries.
    """
    for i in range(retries):
        drone.master.mav.command_long_send(drone.master.target_system, drone.master.target_component, *cmd)
        start = time.time()
        while True:
            msg = drone.get_packet("COMMAND_ACK", no_older_than=1, pop=True, raise_exception=False)
            if msg is None:
                if time.time() - start > timeout:
                    break
                time.sleep(0.2)
                continue
            if msg.command != cmd[0]:
                continue
            if msg.result != 0:
                time.sleep(failed_rebounce)
                break
            print(f"Command {cmd[0]} succeeded")
            return
        print(f"Command {cmd[0]} failed, retrying ({i+1}/{retries})")
    raise CommandException(f"Command {cmd[0]} failed")


def do_command_func(
    drone: Drone, func: Callable, params: List[Any], cmd_code: int, timeout: float = 10, retries: int = 5, failed_rebounce: float = 5
):
    """
    Sends a command via pymavlink then waits for a COMMAND_ACK. If failed it retries.
    """
    for i in range(retries):
        func(*params)
        start = time.time()
        while True:
            msg = drone.get_packet("COMMAND_ACK", no_older_than=1, pop=True, raise_exception=False)
            if msg is None:
                if time.time() - start > timeout:
                    break
                time.sleep(0.2)
                continue
            if msg.command != cmd_code:
                print(f"Received command {msg.command} instead of {cmd_code}")
                continue
            if msg.result != 0:
                time.sleep(failed_rebounce)
                print(f"Command {cmd_code} failed with result {msg.result}")
                break
            print(f"Command {cmd_code} succeeded")
            return
        print(f"Command {cmd_code} failed, retrying ({i+1}/{retries})")
    raise CommandException(f"Command {cmd_code} failed")


def land_at_waypoint_mission(drone: Drone, lat, lon, alt):
    """
    Create and sends a mission to land at the current location with mavlink.
    """
    # clear mission
    drone.master.mav.mission_clear_all_send(
        drone.master.target_system,
        drone.master.target_component,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
    )
    print("Mission clear ACK: ", end="", flush=True)
    for r in range(5):
        res = drone.get_packet("MISSION_ACK", no_older_than=4, pop=True, raise_exception=False)
        if res is None:
            time.sleep(0.3)
            continue
        if res.type:
            raise MissionError("Mission clear failed.")
    print("Success")

    # tell drone to expect 2 mission items
    drone.master.mav.mission_count_send(
        drone.master.target_system,
        drone.master.target_component,
        3,
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
    )
    print("Mission count ACK: ", end="", flush=True)
    res = drone.get_packet("MISSION_REQUEST", no_older_than=4, pop=True)
    print("Success")

    drone.master.mav.mission_item_int_send(
        drone.master.target_system,
        drone.master.target_component,
        0,  # seq
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,  # current - guided-mode request
        1,  # autocontinue
        0,  # p1
        0,  # p2
        0,  # p3
        0,  # p4
        0,  # latitude
        0,  # longitude
        alt,  # altitude
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
    )
    drone.master.mav.mission_item_int_send(
        drone.master.target_system,
        drone.master.target_component,
        1,  # seq
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,  # current - guided-mode request
        1,  # autocontinue
        0,  # p1
        0,  # p2
        0,  # p3
        0,  # p4
        0,  # latitude
        0,  # longitude
        alt,  # altitude
        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
    )
    # Send nav land using MISSION_ITEM_INT
    drone.master.mav.mission_item_int_send(
        drone.master.target_system,
        drone.master.target_component,
        2,  # Sequence number
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,  # The coordinate frame to use
        mavutil.mavlink.MAV_CMD_NAV_LAND,  # The command to send
        0,  # Current WP
        1,  # Autocontinue
        0,
        0,
        0,
        0,  # Params 1-4 (not used for land)
        int(lat * 1e7),  # Latitude in degrees * 1e7
        int(lon * 1e7),  # Longitude in degrees * 1e7
        0,  # Altitude is not used for landing and should be set to 0
    )
    print("Mission send ACK: ", end="", flush=True)
    res = drone.get_packet("MISSION_ACK", no_older_than=4, pop=True)
    if res.type:
        raise MissionError("Mission send failed.")
    print("Success")
