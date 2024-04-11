import os
import cv2
import imdb
import numpy as np
import requests

ia = imdb.IMDb()
print("Loading IMDB database...")


def get_thumbnail_url(movie_name):
    search_movie = ia.search_movie(movie_name)

    if len(search_movie) == 0:
        return None

    url = search_movie[0].data["cover url"]
    try:
        url = url[: url.rindex("@") + 1]
    except ValueError:
        return None
    return url


def download_thumbnail(movie_name, url):
    """
    Download the thumbnail of a movie and returns the relative path with root being the frontend/public folder.
    """
    if url is None:
        return None
    response = requests.get(url)
    if response.status_code == 200:
        filename = movie_name + ".png"
        path = os.path.join(cached_thumbnails_path, filename)
        with open(path, "wb") as file:
            file.write(response.content)
        print("Downloaded thumbnail for " + movie_name + " at " + path)
        return path
    else:
        return None


def check_if_thumbnail_exists(movie_name):
    filename = movie_name + ".png"
    path = os.path.join(cached_thumbnails_path, filename)
    return filename if os.path.exists(path) else None


def create_image_with_label(width, height, label_text):
    image = np.zeros((height, width, 4), dtype=np.uint8)  # Create an image with an alpha channel

    # Determine the dimensions of the label rectangle
    label_font = cv2.FONT_HERSHEY_SIMPLEX
    label_scale = 1.0
    label_thickness = 2
    (label_width, label_height), _ = cv2.getTextSize(label_text, label_font, label_scale, label_thickness)

    # Calculate the position of the label
    label_x = int((width - label_width) / 2)  # Centered horizontally
    label_y = int(0.7 * height)  # 0.7 of the image height

    # Check if the label overflows out of the image
    if label_y + label_height > height:
        label_y = height - label_height  # Adjust the label position

    # Draw the label text on the image with transparent background
    cv2.putText(
        image,
        label_text,
        (label_x, label_y + label_height - 10),
        label_font,
        label_scale,
        (255, 255, 255, 255),
        label_thickness,
        cv2.LINE_AA,
    )

    return image


cached_thumbnails_path = r"frontend/public/cached_thubmnails"
relative_cached_thumbnails_path = r"cached_thubmnails"
temp_images_path = r"frontend/public/temp_images"
relative_temp_images_path = r"temp_images"
