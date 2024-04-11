import threading
import time
import cv2
import numpy as np


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
        def mini_thread():
            while self.flag:
                if tof.data_ready():
                    data = tof.get_data().distance_mm[0][:16]
                    self.forward(data)
                    print("[INFO] MOG background thread running")
                time.sleep(0.1)
            print("[INFO] MOG background thread stopped")

        thread = threading.Thread(target=mini_thread, daemon=True, name="Background")
        thread.start()

    def stop_background(self):
        self.flag = False

    def forward(self, distances_mm):
        background = self.mog.getBackgroundImage()
        numpy_distances = np.array(distances_mm)
        zero_indeces = np.argwhere(numpy_distances == 0)
        data = numpy_distances.astype(np.float64)
        data *= 255.0 / self.distance
        data = np.clip(data, 0, 255).astype(np.uint8)

        # substitute zero values with background values
        if background is not None:
            data[zero_indeces] = np.ravel(background)[zero_indeces]
        # print(list(data))
        mask = self.mog.apply(data)
        if self.display:
            cv2.imshow("data", data.reshape((4, 4)))
            cv2.imshow("mask", mask.reshape((4, 4)))
            cv2.waitKey(1)

        # check if mask is empty
        if not np.all(mask == 0):
            # get indeces of moving pixels
            indeces = self.get_moving_pixels(mask)
            # get values of moving pixels if the value is not 0
            values = [x for x in data[indeces] if 150 > x > 0]

            return values if len(values) > 1 else []
        else:
            return []


if __name__ == "__main__":
    # instantiate tof camera
    tof = tof_utils.tof_setup()

    # instantiate MOG
    mog = MOG()

    while True:
        if tof.data_ready():
            background = mog.getBackgroundImage()
            distance_mm = tof.get_data().distance_mm[0][:16]
            tof_data = np.array(distance_mm)
            # get indeces of zero values
            zero_indeces = np.argwhere(tof_data == 0)
            data = tof_data.astype(np.float64)
            data *= 255.0 / distance
            data = np.clip(data, 0, 255).astype(np.uint8)

            # substitute zero values with background values
            if background is not None:
                data[zero_indeces] = np.ravel(background)[zero_indeces]
            # print(list(data))
            mask = mog.apply(data)

            # check if mask is empty
            if not np.all(mask == 0):
                # get indeces of moving pixels
                indeces = get_moving_pixels(mask)
                # get values of moving pixels if the value is not 0
                values = [x for x in data[indeces] if 150 > x > 0]

                if len(values) > 0:
                    print(f"Movement deteceted at {sum(values) / len(values)}")
                # print(f"Values: {values}, indeces: {indeces}, background: {background_values}")

            # print(list(mask))
            # indeces = get_moving_pixels(mask)

            # make the 16 values into a 4x4 matrix
            # data2 = np.reshape(tof_data, (4, 4))
            cv2.imshow("data", data.reshape((4, 4)))
            cv2.imshow("mask", mask.reshape((4, 4)))
            cv2.waitKey(1)
            # if len(indeces) > 0:
            # try:
            # print(f"Values: {data[indeces]}, indeces: {indeces}, background: {np.ravel(background)[indeces]}")
            # except Exception as e:
            # print(e)
            # print(f"Values: {data[indeces]}, indeces: {indeces}, background: {background}")

            count += 1
            if count == 500:
                print(f"FPS: {count / (time.time() - start)}")
                count = 0
                start = time.time()
                # print(list(mask))
