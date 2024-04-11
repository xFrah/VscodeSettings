import torch
from transformers import pipeline
import cv2
import os
import numpy as np
from PIL import Image
from torch.nn.functional import cosine_similarity


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline(task="image-feature-extraction", model_name="google/vit-base-patch16-384", device=DEVICE, pool=True)


def get_first_and_last_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    first_frame = None
    last_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if first_frame is None:
            first_frame = frame

        last_frame = frame

    cap.release()
    return first_frame, last_frame


def get_video_similarity(video_path):
    print("Processing video:", video_path)

    reference_image, given_image = get_first_and_last_frames(video_path)
    if reference_image is None or given_image is None:
        print("Error: Video file not found or is empty.")
        return

    # convertto pil image
    reference_image = Image.fromarray(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))

    outputs = pipe([reference_image, given_image])

    similarity_score = cosine_similarity(torch.Tensor(outputs[0]), torch.Tensor(outputs[1]), dim=1)
    print(similarity_score)

    print(np.array(outputs).shape)

    # # resize to same size
    # reference_image = cv2.resize(reference_image, (1280, 720))
    # given_image = cv2.resize(given_image, (1280, 720))

    # # convert to numpy array
    # reference_image = np.array(reference_image)
    # given_image = np.array(given_image)

    # # convert to PIL
    # reference_image = Image.fromarray(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
    # given_image = Image.fromarray(cv2.cvtColor(given_image, cv2.COLOR_BGR2RGB))

    # # finding difference
    # diff = ImageChops.difference(reference_image, given_image)

    # showing the difference
    # diff.show()
    # also show the reference and given image
    reference_image.show()
    given_image.show()


if __name__ == "__main__":
    # get all videos in CACHE directory
    cache_dir = "CACHE"
    video_files = [f for f in os.listdir(cache_dir) if f.endswith(".mp4") or f.endswith(".265")]

    for video_file in video_files:
        get_video_similarity(os.path.join("CACHE", video_file))
