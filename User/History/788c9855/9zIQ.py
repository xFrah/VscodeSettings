import os
import imdb
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
    if url is None:
        return None
    response = requests.get(url)
    if response.status_code == 200:
        filename = movie_name + ".png"
        path = os.path.join("frontend/public", filename)
        with open(path, "wb") as file:
            file.write(response.content)
        return filename
    else:
        return None


def check_if_thumbnail_exists(movie_name):
    filename = movie_name + ".png"
    path = os.path.join("frontend/public", filename)
    return filename if os.path.exists(path) else None
