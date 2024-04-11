import threading
import time
import cv2
import numpy as np
import tof_utils


class MOG:
    def __init__(self, distance_threshold=150, var_threshold=150, detect_shadows=False, display=False):
        self.mog = cv2.createBackgroundSubtractorMOG2(detectShadows=detect_shadows, varThreshold=var_threshold)
        self.var_threshold = self.mog.getVarThreshold()
        self.history = self.mog.getHistory()
        self.distance_threshold = distance_threshold
        self.distance = 512
        self.flag = True
        self.display = display

    def get_moving_pixels(self, matrix):
        indices = np.argwhere(matrix > 128)
        return [idx[0] for idx in indices]

    def set_history(self, history):
        self.history = history

    def start_background(self, tof):
        self.flag = True

        def mini_thread():
            while self.flag:
                if tof.data_ready():
                    data = tof.get_data().distance_mm[0]
                    self.forward(data)
                time.sleep(0.1)
            print("[INFO] MOG background thread stopped")

        thread = threading.Thread(target=mini_thread, daemon=True, name="Background")
        thread.start()

    def stop_background(self):
        self.flag = False

    def forward(self, distances_mm, display=None):
        start = time.time()
        background = self.mog.getBackgroundImage()
        numpy_distances = np.array(distances_mm)
        zero_indeces = np.argwhere(numpy_distances == 0)
        data = numpy_distances.astype(np.float64)
        data *= 255.0 / self.distance
        data = np.clip(data, 0, 255).astype(np.uint8)
        print("[INFO] MOG 1 took {:.2f} seconds".format(time.time() - start))

        # substitute zero values with background values
        if background is not None:
            start = time.time()
            data[zero_indeces] = np.ravel(background)[zero_indeces]
            print("[INFO] MOG 2 took {:.2f} seconds".format(time.time() - start))
        # print(list(data))
        start = time.time()
        mask = self.mog.apply(data)
        print("[INFO] MOG 3 took {:.2f} seconds".format(time.time() - start))
        if self.display or display:
            start = time.time()
            cv2.imshow("data", data.reshape((8, 8)))
            cv2.imshow("mask", mask.reshape((8, 8)))
            cv2.waitKey(1)
            print("[INFO] MOG 4 took {:.2f} seconds".format(time.time() - start))

        start = time.time()
        # check if mask is empty
        if not np.all(mask == 0):
            # get indeces of moving pixels
            indeces = self.get_moving_pixels(mask)
            # get values of moving pixels if the value is not 0
            values = [x for x in data[indeces] if 150 > x > 0]
            print("[INFO] MOG 5 took {} ms".format(time.time() - start))
            return values if len(values) > 3 else []
        else:
            return []


if __name__ == "__main__":
    count = 0
    start = time.time()
    # instantiate tof camera
    tof = tof_utils.tof_setup()

    # instantiate MOG
    mog = MOG()

    while True:
        if tof.data_ready():
            distance_mm = tof.get_data().distance_mm[0]
            values = mog.forward(distance_mm, display=True)
            print(values)
            count += 1
            if count == 50:
                print(f"FPS: {count / (time.time() - start)}")
                count = 0
                start = time.time()
                # print(list(mask))
