import time
import cv2
import numpy as np


class Pentagramma:
    def __init__(self):
        self.index = 0
        self.line_thickness = 1
        self.height = 100
        self.width = 300
        self.interspace = self.height // 12
        self.offset = self.interspace
        self.notes = {"sol": 0, "fa": 1, "mi": 2, "re": 3, "do": 4, "si": 5, "la": 6, "sol2": 7, "fa2": 8, "mi2": 9, "re2": 10, "do2": 11}
        self.max_notes_on_screen = 10
        self.note_offset = self.width // (self.max_notes_on_screen + 1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < 10:
            self.index += 1
            return self.index
        else:
            raise StopIteration

    def get_base(self):
        # create white image of height self.height and width self.width with numpy
        img = np.ones((self.height, self.width, 3), np.uint8) * 255
        for i in range(11):
            # print line with thickness self.line_thickness and y offset self.interspace
            y = self.offset + (self.interspace * i)
            if i % 2 == 0:
                # cv2.line(img, (0, y), (self.width, y), (0, 0, 0), self.line_thickness)
                pass
            else:
                cv2.line(img, (0, y), (self.width, y), (0, 0, 0), self.line_thickness)
        self.draw_note("sol", img)
        return img

    def draw_note(self, note, img):
        # get y coordinate of note
        y = self.offset + (self.interspace * self.notes[note])
        # draw note
        cv2.ellipse(img, (self.width // 2, y), (int(self.interspace * 1.2), int(self.interspace / 1.2)), 0, 0, 360, (0, 0, 0), -1)
        return img


def main():
    pentagramma = Pentagramma()
    cv2.imshow("pentagramma", pentagramma.get_base())
    cv2.waitKey(0)
    time.sleep(5)


main()
