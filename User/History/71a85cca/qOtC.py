import time
import cv2
import numpy as np
from pitch import Pitcher


class Pentagramma:
    def __init__(self):
        self.line_thickness = 1
        self.height = 100
        self.width = 60
        self.interspace = self.height // 12
        self.offset = self.interspace
        self.notes = {"sol": 0, "fa": 1, "mi": 2, "re": 3, "do": 4, "si": 5, "la": 6}  # "sol2": 7, "fa2": 8, "mi2": 9, "re2": 10}
        self.max_notes_on_screen = 6
        self.x_offset = 30
        self.note_offset = (self.width - (self.x_offset * 2)) // (self.max_notes_on_screen + 1)

    def __iter__(self):
        return self

    def __next__(self):
        random_note = np.random.choice(list(self.notes.keys()))
        print(random_note)
        new_base = self.get_base()
        self.draw_note(random_note, new_base)
        return new_base

    def get_base(self):
        # create white image of height self.height and width self.width with numpy
        img = np.ones((self.height, self.width, 3), np.uint8) * 255
        for i in range(11):
            # print line with thickness self.line_thickness and y offset self.interspace
            y = self.offset + (self.interspace * i)
            if i % 2 != 0:
                cv2.line(img, (0, y), (self.width, y), (0, 0, 0), self.line_thickness)
        return img

    def draw_note(self, note, img):
        # get y coordinate of note
        y = self.offset + (self.interspace * self.notes[note])
        x = self.x_offset
        # draw note
        cv2.ellipse(img, (x, y), (int(self.interspace * 1.2), int(self.interspace / 1.2)), 0, 0, 360, (0, 0, 0), -1)
        return img


def slide_animation(img, prev, remove_first=False):
    frames = []
    for column in range(img.shape[1]):
        # delete first line of prev and add the first line from img to the right
        prev_ = np.hstack((prev[:, column:] if remove_first else prev, img[:, :column]))
        frames.append(prev_)
    return frames


def main():
    pentagramma = Pentagramma()
    pitcher = Pitcher()

    # create image with just one white column
    prev = np.ones((pentagramma.height, 1, 3), np.uint8) * 255
    c = 0
    for img in pentagramma:
        c += 1
        # create animation by concatenating 1 column per loop iteration and deleting the first one
        frames = slide_animation(img, prev, remove_first=False if c < 6 else True)
        for frame in frames:
            cv2.imshow("pentagramma", frame)
            cv2.waitKey(1)
            time.sleep(0.01)
        prev = frames[-1]
        while True:
            pitch = pitcher.get_current_note()
            if pitch is not None:
                temp = prev.copy()
                pentagramma.draw_note(pitch, temp)
                cv2.imshow("pentagramma", temp)
            time.sleep(0.1)


main()
