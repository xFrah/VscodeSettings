import pyaudio
import crepe
import os
import numpy as np
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide TF logging


class Pitcher:
    def __init__(self):
        self.CHUNK = 1024  # Number of samples per buffer
        self.WIDTH = 2  # Bytes per sample
        self.CHANNELS = 1  # Mono sound
        self.RATE = 16000  # Sampling rate (number of samples per second)

    def _thread(self):
        try:
            at = []

            print("Recording is starting...")
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(WIDTH), channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            while True:
                st = time.time()

                bytes_data = stream.read(CHUNK)
                nparray_data = np.frombuffer(bytes_data, dtype=np.int16)  # Convert bytes to NumPy ndarray

                time_f, frequency, confidence, activation = crepe.predict(nparray_data, RATE, model_capacity="tiny", step_size=65, verbose=0)
                # Console outcome
                confidence_mark = "🟥"
                if confidence[0] >= 0.4:
                    confidence_mark = "🟩"
                print(f"{confidence_mark} {round(frequency[0], 1)} Hz | {confidence[0]}")

                at.append(time.time() - st)
        except KeyboardInterrupt:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Recording stopped")

            at.pop(0)
            print(f"Average time per frame: {round(sum(at) / len(at), 3)} sec.")

    # close thread when object is deleted
    def __del__(self):
        self._thread.join()
