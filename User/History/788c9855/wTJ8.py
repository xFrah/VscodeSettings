import os
import cv2
import imdb
import numpy as np
import requests
import urllib.parse
import textwrap

ia = imdb.IMDb()
print("[TU] Loading IMDB database...")


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
        relative = os.path.join(relative_cached_thumbnails_path, filename)
        with open(path, "wb") as file:
            file.write(response.content)
        print("[TU] Downloaded thumbnail for " + movie_name + " at " + path)
        print("[TU] Relative path: " + relative)
        return relative
    else:
        return None


def check_if_thumbnail_exists(movie_name):
    filename = movie_name + ".png"
    path = os.path.join(cached_thumbnails_path, filename)
    relative = os.path.join(relative_cached_thumbnails_path, filename)
    return relative if os.path.exists(path) else None


def url_to_file_name(url):
    parsed_url = urllib.parse.urlparse(url)
    file_name = urllib.parse.unquote(parsed_url.path)  # Decode URL
    file_name = file_name.split("/")[-1]  # Extract the file name from the path
    file_name = os.path.splitext(file_name)[0]  # Remove the extension
    return file_name


def create_image_with_label(label_text):
    # get unknown.png
    img = cv2.imread("unknown.png", cv2.IMREAD_UNCHANGED)

    wrapped_text = textwrap.wrap(label_text, width=30)
    font_size = 1
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, line in enumerate(wrapped_text):
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]

        gap = textsize[1] + 10

        y = (int((img.shape[0] + textsize[1]) / 2) + i * gap) + 100
        x = int((img.shape[1] - textsize[0]) / 2)

        cv2.putText(img, line, (x, y), font, font_size, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

    return img


cached_thumbnails_path = r"frontend/public/cached_thubmnails"
relative_cached_thumbnails_path = r"cached_thubmnails"
temp_images_path = r"frontend/public/temp_images"
relative_temp_images_path = r"temp_images"

# create image and show it
if __name__ == "__main__":
    cv2.imshow("image", create_image_with_label("Hello Worldkjnskjvnfskdjfnkjndfkjdsnfksdjnfkjdsnfkjdsnfkdsjnfkjdsfn"))
    cv2.waitKey(0)
