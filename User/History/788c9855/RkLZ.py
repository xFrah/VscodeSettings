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
    response = requests.get(url)
    if response.status_code == 200:
        path = os.path.join("frontend/public", movie_name, ".png")
        with open(path, "wb") as file:
            file.write(response.content)
        return path
    else:
        return None


def check_if_thumbnail_exists(movie_name):
    path = os.path.join("frontend/public", movie_name, ".png")
    return path if os.path.exists(path) else None
