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
    
    def get_cropped_and_translated_image3(self, image, width):
        original_height, original_width = image.shape[:2]
        # get image in roi and fill in blank image
        roi_polygon_adjusted = [(x * original_width, y * original_height) for x, y in self.roi_polygon]
        roi_polygon_adjusted = np.array(roi_polygon_adjusted, np.float32)
        roi_polygon_adjusted = roi_polygon_adjusted.reshape((-1, 1, 2))

        # map image in roi to blank image
        M = cv2.getPerspectiveTransform(roi_polygon_adjusted, np.array([[0, 0], [width, 0], [width, 256], [0, 256]], np.float32))
        warped_image = cv2.warpPerspective(image, M, (width, 256))

        # get the cropped image
        cropped_image = warped_image[0:256, 0:width]

        return cropped_image

    def get_cropped_image2(self, image):
        # Get original image dimensions and aspect ratio
        original_height, original_width = image.shape[:2]

        # Calculate the bounding box of the ROI
        xmin, ymin, xmax, ymax = self.get_bounding_box()
        # convert to real pixels from frac
        xmin, ymin, xmax, ymax = int(xmin * original_width), int(ymin * original_height), int(xmax * original_width), int(ymax * original_height)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        roi_polygon_adjusted = [(x * original_width, y * original_height) for x, y in self.roi_polygon]
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

def postprocess(preds, img, orig_img, OBJ_THRESH, NMS_THRESH, classes=None):
    p = non_max_suppression(preds[0],
                                OBJ_THRESH,
                                NMS_THRESH,
                                agnostic=False,
                                max_det=300,
                                nc=classes,
                                classes=None)        
    results = []
    for i, pred in enumerate(p):
        shape = orig_img.shape
        if not len(pred):
            results.append([[], []])  # save empty boxes
            continue
        pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        results.append([pred[:, :6], shape[:2]])
    return results


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def preprocess(image, input_height, input_width):
    image_3c = image

    # Convert the image_3c color space from BGR to RGB
    image_3c = cv2.cvtColor(image_3c, cv2.COLOR_BGR2RGB)

    # Resize the image_3c to match the input shape
    # image_3c = cv2.resize(image_3c, (input_width, input_height))
    image_3c, ratio, dwdh = letterbox(image_3c, new_shape=[input_height, input_width], auto=False)

    # Normalize the image_3c data by dividing it by 255.0
    image_4c = np.array(image_3c) / 255.0

    # Transpose the image_3c to have the channel dimension as the first dimension
    image_4c = np.transpose(image_4c, (2, 0, 1))  # Channel first

    # Expand the dimensions of the image_3c data to match the expected input shape
    image_4c = np.expand_dims(image_4c, axis=0).astype(np.float32)

    image_4c = np.ascontiguousarray(image_4c)  # contiguous

    # Return the preprocessed image_3c data
    return image_4c, image_3c

