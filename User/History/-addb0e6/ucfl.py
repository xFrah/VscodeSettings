from abc import ABC, abstractmethod
import time
from typing import Tuple
import threading
from drone_helpers import (
    Drone,
    arm,
    change_mode,
    distance_to_waypoint,
    get_mode,
    land_at_waypoint_mission,
    progress_debug,
)


class WaypointFailedException(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


class Waypoint(ABC):
    def __init__(self, lat: float, lon: float, finished_condition_polling=1) -> None:
        self.lat = lat
        self.lon = lon
        self.drone = None
        self.failed = False
        self.ran = False
        self.finished = False
        self.finished_condition_polling = finished_condition_polling

    def __call__(self, drone: Drone) -> None:
        self.drone = drone

        def threaded_run():
            try:
                self._run()
                self.ran = True
            except Exception as e:
                print(f"{self} failed with error: {e}")
                self.__set_failed()

        def threaded_finished_condition():
            try:
                while True:  # update waypoint status every second
                    time.sleep(1)
                    if not self.ran:  # waypoint has not finished running
                        continue
                    finished = self._finished_condition()
                    if finished:
                        self.finished = True
                        break
            except Exception as e:
                print(f"{self} failed with error: {e}")
                self.__set_failed()

        threading.Thread(target=threaded_run, daemon=True).start()
        threading.Thread(target=threaded_finished_condition, daemon=True).start()

    @abstractmethod
    def _run(self) -> None:
        """
        Implement the waypoint here.
        This will be run in a separate thread, so don't worry about blocking.
        """
        pass

    @abstractmethod
    def _finished_condition(self) -> bool:
        """
        Should return True if the waypoint has finished, False otherwise.
        This always gets called after run() has completed.
        """
        pass

    @abstractmethod
    def _failsafe(self) -> None:
        """
        Method to be executed if the waypoint fails.
        """
        pass

    def wait(self) -> bool:
        """
        Waits for the waypoint to finish.
        Returns True if the waypoint succeeded, False otherwise.
        """
        while True:
            finished, failed = self.has_finished()
            if finished:
                if failed:
                    return False
                return True
            time.sleep(self.finished_condition_polling)

    def __set_failed(self) -> None:
        """
        Sets the waypoint as failed and executes failsafe.
        """
        try:
            self._failsafe()
        except Exception as e:
            print(f"{self} failed to execute failsafe: {e}")
        self.failed, self.finished = True, True

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.lat:.06f}, {self.lon:.06f})"

    def has_finished(self) -> Tuple[bool, bool]:
        """
        Returns a tuple of two booleans:
        - has_finished
        - has_failed
        """
        if self.drone is None:  # waypoint has not been run yet
            return False, False
        if self.failed:  # waypoint has failed
            return True, True
        if not self.ran:  # waypoint has not finished running
            return False, False
        if self.finished:  # waypoint has finished
            return True, False
        return False, False


class LandWaypoint(Waypoint):
    def __init__(self, lat: float, lon: float, alt_target: int = 15, debug=False):
        """
        Land at specified coordinates.
        """
        self.alt_target = alt_target  # TODO can we make this a float?
        self.debug = debug
        super().__init__(lat, lon)

    def _run(self):
        self.drone.change_mode("GUIDED")
        self.drone.arm()
        # TODO make commands more general and use them here
        land_at_waypoint_mission(self.drone, self.lat, self.lon, self.alt_target)
        self.drone.change_mode("AUTO")  # start mission

    def _finished_condition(self):
        # TODO add stale timeout (check if vehicle is moving)
        if (mode := self.drone.mode) != "AUTO":
            raise WaypointFailedException(
                f"Vehicle is not in AUTO mode. ({mode} != AUTO)"
            )
        distance, (_, _, alt) = self.drone.distance_to_waypoint(self.lat, self.lon)
        if not self.drone.is_armed and distance >= 2:
            raise WaypointFailedException("Vehicle landed before reaching waypoint.")
        if self.debug:
            progress_debug(self.alt_target, distance, alt)
        return distance <= 1 and not self.drone.is_armed

    def _failsafe(self):
        self.drone.change_mode("RTL")
