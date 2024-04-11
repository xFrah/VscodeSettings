# check if (x, y) is between vertices
def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# object classes
classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

import csv
import cv2
import numpy as np


class ROI:
    def __init__(self, roi_path):
        self.roi_polygon = []
        with open(roi_path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                self.roi_polygon.append((int(row[0]), int(row[1])))
        print("ROI polygon:", self.roi_polygon)

    def is_inside(self, x, y):
        n = len(self.roi_polygon)
        inside = False
        p1x, p1y = self.roi_polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.roi_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def get_bounding_box(self):
        xs = [point[0] for point in self.roi_polygon]
        ys = [point[1] for point in self.roi_polygon]
        return min(xs), min(ys), max(xs), max(ys)

    def get_masked_image(self, image):
        # Assuming roi_polygon is normalized ([0, 1] range) and needs to be scaled
        height, width = image.shape[:2]
        scaled_roi_polygon = [(int(x * width), int(y * height)) for x, y in self.roi_polygon]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        roi_polygon = np.array(scaled_roi_polygon, np.int32)
        roi_polygon = roi_polygon.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [roi_polygon], 255)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image


    def get_cropped_image(self, image):
        # Get original image dimensions and aspect ratio
        original_height, original_width = image.shape[:2]
        original_aspect_ratio = original_width / original_height

        # Calculate the bounding box of the ROI
        xmin, ymin, xmax, ymax = self.get_bounding_box()
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

        # Crop the image using integer indices
        cropped_image = image[ymin:ymax, xmin:xmax]

        # Adjusted mask creation for the cropped image to use integer coordinates
        mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
        roi_polygon_adjusted = [(int(x) - xmin, int(y) - ymin) for x, y in self.roi_polygon]
        roi_polygon_adjusted = np.array(roi_polygon_adjusted, np.int32)
        roi_polygon_adjusted = roi_polygon_adjusted.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [roi_polygon_adjusted], 255)

        # Apply the mask
        masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
        return masked_image

    def fit_to_resolution(self, image, target_width, target_height):
        # Crop the image to the bounding box of the ROI
        xmin, ymin, xmax, ymax = self.get_bounding_box()
        cropped_image = image[int(ymin) : int(ymax), int(xmin) : int(xmax)]

        # Calculate the aspect ratio of the target resolution
        target_aspect_ratio = target_width / target_height

        # Calculate the aspect ratio of the cropped image
        cropped_height, cropped_width = cropped_image.shape[:2]
        cropped_aspect_ratio = cropped_width / cropped_height

        # Determine new dimensions based on aspect ratios
        if cropped_aspect_ratio > target_aspect_ratio:
            # Cropped image is wider than target aspect
            new_width = target_width
            new_height = int(target_width / cropped_aspect_ratio)
        else:
            # Cropped image is taller than target aspect
            new_height = target_height
            new_width = int(target_height * cropped_aspect_ratio)

        # Resize the cropped image
        resized_image = cv2.resize(cropped_image, (new_width, new_height))

        # Create padding if necessary
        delta_width = target_width - new_width
        delta_height = target_height - new_height
        top, bottom = delta_height // 2, delta_height - (delta_height // 2)
        left, right = delta_width // 2, delta_width - (delta_width // 2)

        # Apply padding
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        return padded_image


def delete_folder(folder):
    import os
    import shutil

    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted {folder}")
    else:
        print(f"{folder} does not exist")
