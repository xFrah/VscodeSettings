import time
import cv2
import numpy as np


class Pentagramma:
    def __init__(self):
        self.line_thickness = 1
        self.height = 100
        self.width = 300
        self.interspace = self.height // 12
        self.offset = self.interspace
        self.notes = {"sol": 0, "fa": 1, "mi": 2, "re": 3, "do": 4, "si": 5, "la": 6, "sol2": 7, "fa2": 8, "mi2": 9, "re2": 10, "do2": 11}
        self.max_notes_on_screen = 10
        self.notes_on_screen = []
        self.x_offset = 10
        self.note_offset = (self.width - (self.x_offset * 2)) // (self.max_notes_on_screen + 1)

    def __iter__(self):
        return self

    def __next__(self):
        random_note = np.random.choice(list(self.notes.keys()))
        print(random_note)
        self.notes_on_screen.append(random_note)
        del self.notes_on_screen[: -self.max_notes_on_screen]
        new_base = self.get_base()
        for i, note in enumerate(self.notes_on_screen):
            self.draw_note(note, new_base, i)
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

    def draw_note(self, note, img, index):
        # get y coordinate of note
        y = self.offset + (self.interspace * self.notes[note])
        x = self.x_offset + (self.note_offset * index)
        # draw note
        cv2.ellipse(img, (x, y), (int(self.interspace * 1.2), int(self.interspace / 1.2)), 0, 0, 360, (0, 0, 0), -1)
        return img


def main():
    pentagramma = Pentagramma()
    for img in pentagramma:
        cv2.imshow("pentagramma", img)
        cv2.waitKey(1)
        time.sleep(1)


main()
