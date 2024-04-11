import threading
import time

import serial


class Lidar:
    def __init__(self, port, name=None, slices=False, buffer_size=5):
        self.name = name
        self.port = port
        self.serial_lidar: serial.Serial = serial.Serial(self.port, 460800)
        self._read_thread: threading.Thread = threading.Thread(target=self._read_data)
        self.buffer_size = buffer_size
        self.data = bytes()
        self.point_clouds = []
        self.reading = False
        self.slices = []
        self.slices_bool = slices
        self.slice_medium_angles = {
            i: angle
            for i, angle in enumerate(
                sorted(
                    [
                        270.3214274,
                        285.3214273999999,
                        300.3214273999999,
                        315.3214274,
                        330.32142740000006,
                        345.32142740000006,
                        360.3214274,
                        375.32142740000006,
                        30.321427399999997,
                        45.321427400000005,
                        60.32142739999999,
                        75.3214274,
                        90.32142740000002,
                        105.32142740000002,
                        120.3214274,
                        135.3214274,
                        150.32142740000003,
                        165.32142739999998,
                        180.3214274,
                        195.32142739999995,
                        210.3214274,
                        225.32142739999995,
                        240.32142739999995,
                        255.32142739999995,
                    ]
                )
            )
        }
        self._read_thread.start()

    def decode(self, line):
        try:
            if line[0] << 8 | line[1] != 0xA55A:
                print("Lidar >> Invalid header")
                return

            # angle at the start of the packet
            angle = (line[2] << 8 | line[3]) // 100
            # list of tuples (angle, distance), each angle differs by 0.3571 degrees
            distances = [(angle + (0.3571428 * ((i - 6) / 2)), line[i] << 8 | line[i + 1]) for i in range(6, 89, 2)]
            return distances
        except IndexError:
            print("Lidar >> Invalid line")
            return

    def _read_data(self):
        current_point_cloud = []
        slice_counter = 0
        slices_buffer = []
        last = time.time()
        while True:
            to_add = self.serial_lidar.read_all()
            if (to_add and len(to_add) == 0) or to_add is None:
                time.sleep(0.1)
                print("Lidar >> No data")
                if last < time.time() - 5:
                    self.reload_serial_connection()
                    last = time.time()
                    print("Lidar >> Successfully reset")
                continue
            last = time.time()
            self.data += to_add

            # search for the first full packet in the buffer
            start = self.data.find(b"\xa5\x5a")
            if start == -1:
                self.data = bytes()
                print("Lidar >> Header not found")
            else:

                # we cut off the data before the first packet
                self.data = self.data[start:]

                # then we search for the tail
                end = self.data.find(b"\xfa\xfb")
                while end != -1:

                    # we know that after every tail there is another header, so we keep searching until there is no more packets
                    slice_counter += 1
                    data_to_decode = self.data[: end + 2]
                    slice_ = self.decode(data_to_decode)
                    if slice_ is None:
                        print("Lidar >> Invalid slice")
                        self.data = self.data[end + 2 :]
                        break
                    if self.slices_bool:
                        slices_buffer.append(slice_)
                    current_point_cloud += slice_

                    # when we get the full point cloud, we deliver it outside
                    if slice_counter == 24:
                        slice_counter = 0
                        if self.reading:
                            self.point_clouds.append(current_point_cloud)
                            if self.slices_bool:
                                self.slices.append(slices_buffer)
                                slices_buffer = []
                        current_point_cloud = []

                        # cutting off the buffers if they are too big
                        self.slices[: -self.buffer_size] = []
                        self.point_clouds[: -self.buffer_size] = []

                    self.data = self.data[end + 2 :]
                    end = self.data.find(b"\xfa\xfb")
            time.sleep(0.01)

    def get_slice_at_angle(self, angle):
        return min(self.slice_medium_angles.items(), key=lambda tuple_: abs(tuple_[1] - angle))[0]

    def get_distance_at_angle(self, slices, angle):
        try:
            slices = sorted(slices, key=lambda x: x[0][0])
            slice_index = self.get_slice_at_angle(angle)
            return min([x for x in slices[slice_index] if x[1] > 30], key=lambda x: x[1])[1], slice_index
        except Exception as e:
            print(e)
            return None, None

    def get_distances(self, pop=True):
        # print("Lidar >> Getting point cloud, buffer size: ", len(self.point_clouds))
        return (list.pop if pop else list.__getitem__)(self.point_clouds, 0) if len(self.point_clouds) else None

    def get_slices(self, pop=True):
        # print("Lidar >> Getting slices, buffer size: ", len(self.slices))
        return (list.pop if pop else list.__getitem__)(self.slices, 0) if len(self.slices) else None

    def fix_buffer_leak(self):
        if len(self.point_clouds) > self.buffer_size or len(self.slices) > self.buffer_size:
            self.slices[: -self.buffer_size] = []
            self.point_clouds[: -self.buffer_size] = []

    def test_get_medium_angles(self):
        slice_ = self.get_slices()
        if slice_ is None:
            return None
        # return a list of medium angles for each slice in slice_
        return [sum([angle for angle, _ in slice_[i]]) / len(slice_[i]) for i in range(len(slice_))]

    def active(self, is_reading: bool):
        if not isinstance(is_reading, bool):
            return print("Lidar >> Invalid type, active takes a bool as argument.")
        self.reading = is_reading

    def reload_serial_connection(self):
        self.serial_lidar.close()
        self.serial_lidar = serial.Serial(self.port, 460800)

    def get_max_and_min_angles(self):
        return min(self.get_distances(), key=lambda x: x[0])[0], max(self.get_distances(), key=lambda x: x[0])[0]


if __name__ == "__main__":
    lidar = Lidar(slices=True)
    lidar.active(True)
    time.sleep(2)
    print(lidar.test_get_medium_angles())
    print(lidar.get_max_and_min_angles())
