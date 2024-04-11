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
                self.roi_polygon.append((float(row[0]), float(row[1])))
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

    def get_scaled_roi_polygon(self, image):
        height, width = image.shape[:2]
        return [(int(x * width), int(y * height)) for x, y in self.roi_polygon]

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

    def get_cropped_image2(self, image):
        # Get original image dimensions and aspect ratio
        original_height, original_width = image.shape[:2]

        # Calculate the bounding box of the ROI
        xmin, ymin, xmax, ymax = self.get_bounding_box()
        # convert to real pixels from frac
        xmin, ymin, xmax, ymax = int(xmin * original_width), int(ymin * original_height), int(xmax * original_width), int(ymax * original_height)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        roi_polygon_adjusted = [((x * original_width) - int(xmin), (y * original_height) - int(ymin)) for x, y in self.roi_polygon]
        roi_polygon_adjusted = np.array(roi_polygon_adjusted, np.int32)
        roi_polygon_adjusted = roi_polygon_adjusted.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [roi_polygon_adjusted], 255)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # get the cropped image
        cropped_image = masked_image[ymin:ymax, xmin:xmax]
        return cropped_image

    def get_cropped_image(self, image):
        # Get original image dimensions and aspect ratio
        original_height, original_width = image.shape[:2]
        original_aspect_ratio = original_width / original_height

        # Calculate the bounding box of the ROI
        xmin, ymin, xmax, ymax = self.get_bounding_box()
        # convert to real pixels from frac
        xmin, ymin, xmax, ymax = int(xmin * original_width), int(ymin * original_height), int(xmax * original_width), int(ymax * original_height)
        roi_width, roi_height = xmax - xmin, ymax - ymin

        # Adjust the bounding box to maintain the original aspect ratio
        # Check which dimension (width or height) needs to be adjusted
        if roi_width / roi_height > original_aspect_ratio:
            # Width is larger relative to height, adjust height to match aspect ratio
            adjusted_height = roi_width / original_aspect_ratio
            ymin = max(0, ymin - (adjusted_height - roi_height) / 2)
            ymax = ymin + adjusted_height
        else:
            # Height is larger relative to width, adjust width to match aspect ratio
            adjusted_width = roi_height * original_aspect_ratio
            xmin = max(0, xmin - (adjusted_width - roi_width) / 2)
            xmax = xmin + adjusted_width

        # Ensure the bounding box is within the image dimensions
        xmin, xmax = max(0, xmin), min(original_width, xmax)
        ymin, ymax = max(0, ymin), min(original_height, ymax)

        # Crop the image to the adjusted bounding box
        cropped_image = image[int(ymin) : int(ymax), int(xmin) : int(xmax)]

        # Create a mask for the ROI within the cropped image
        mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
        roi_polygon_adjusted = [((x * original_width) - int(xmin), (y * original_height) - int(ymin)) for x, y in self.roi_polygon]
        roi_polygon_adjusted = np.array(roi_polygon_adjusted, np.int32)
        roi_polygon_adjusted = roi_polygon_adjusted.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [roi_polygon_adjusted], 255)

        # Apply the mask to the cropped image
        masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

        return masked_image


def delete_folder(folder):
    import os
    import shutil

    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Deleted {folder}")
    else:
        print(f"{folder} does not exist")
