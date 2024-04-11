import threading
import pyaudio
import librosa
import crepe
import os
import numpy as np
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide TF logging


class Pitcher:
    def __init__(self, CHUNK=1024, WIDTH=2, CHANNELS=1, RATE=16000):
        self.CHUNK = CHUNK  # Number of samples per buffer
        self.WIDTH = WIDTH  # Bytes per sample
        self.CHANNELS = CHANNELS  # Mono sound
        self.RATE = RATE  # Sampling rate (number of samples per second)
        self._running = True
        self.thread = threading.Thread(target=self._thread)
        self.thread.start()
        self.good = {}
        self.last_check = time.time()

    def _garbage_collector(self):
        now = time.time()
        for k in list(self.good):
            if now - k >= 0.5:
                del self.good[k]

    def get_pitch(self):
        self._garbage_collector()
        self.last_check = time.time()
        if len(self.good) == 0:
            return None
        # get last three values
        values = list(self.good.values())[-3:]
        return round(sum(values) / len(values), 1)

    def get_current_note(self):
        self._garbage_collector()
        if len(self.good) == 0:
            return None
        return librosa.hz_to_note(sum(self.good.values()) / len(self.good.values()), octave=False)

    def _thread(self):
        try:
            at = []

            print("Recording is starting...")
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(self.WIDTH), channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK
            )
            while self._running:
                st = time.time()

                bytes_data = stream.read(self.CHUNK)
                nparray_data = np.frombuffer(bytes_data, dtype=np.int16)  # Convert bytes to NumPy ndarray

                time_f, frequency, confidence, activation = crepe.predict(
                    nparray_data, self.RATE, model_capacity="tiny", step_size=65, verbose=0
                )
                # Console outcome
                confidence_mark = "🟥"
                if confidence[0] >= 0.4:
                    confidence_mark = "🟩"
                print(f"{confidence_mark} {round(frequency[0], 1)} Hz | {confidence[0]}")

                # Store good values
                if confidence[0] >= 0.4:
                    self.good[time.time()] = frequency[0]

                # get note from frequency

                at.append(time.time() - st)

                if time.time() - self.last_check > 0.2:
                    self.last_check = time.time()
        except KeyboardInterrupt:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Recording stopped")

            at.pop(0)
            print(f"Average time per frame: {round(sum(at) / len(at), 3)} sec.")

    # close thread when object is deleted
    def __del__(self):
        self._running = False
        self._thread.join()
