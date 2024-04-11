import json
import os
from audio_utils import get_audio_tracks_info
from thumbnail_utils import get_thumbnail_url, download_thumbnail, check_if_thumbnail_exists, url_to_file_name


class Movie:
    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.path = path
        print(f"MU >> Movie path for {self.name}: {self.path}")
        self.audio_tracks = get_audio_tracks_info(path)
        self.cached_thumbnail = self.cache_thumbnail()
        self.track_to_interface = {}
        self.config_file = "filmconfig_" + os.path.splitext(self.name)[0] + ".json"

    def get_config(self) -> dict[str, str]:
        """
        Load the configuration for the movie.
        """
        # check if the config file exists
        if os.path.exists(self.config_file):
            # load the dictionary
            with open(self.config_file, "r") as f:
                return json.load(f)
        else:
            print("MU >> Movie " + self.name + " does not have a config file at '" + self.config_file + "'.")
            return {}

    def cache_thumbnail(self, force: bool = False):
        """
        Cache the thumbnail for the movie.
        """
        if force:
            return download_thumbnail(self.name, get_thumbnail_url(self.name))
        else:
            path = check_if_thumbnail_exists(self.name)
            return path if path is not None else download_thumbnail(self.name, get_thumbnail_url(self.name))

    def save_config(self, track_to_guid: dict[int, str]):
        """
        Save the configuration for the movie.
        """
        # dump the dictionary
        with open(self.config_file, "w") as f:
            json.dump(track_to_guid, f)

    def has_config(self):
        """
        Check if the movie has a configuration file.
        """
        has_it = os.path.exists(self.config_file)
        if has_it:
            print("MU >> Movie " + self.name + " has a config file.")
        else:
            print("MU >> Movie " + self.name + " does not have a config file at '" + self.config_file + "'.")
        return has_it

    def get_audio_tracks(self):
        """
        Get the audio tracks of the movie.
        """
        return self.audio_tracks


class Movies:
    def __init__(self, folder: str) -> None:
        if not os.path.exists(cached_thumbnails_path):
            os.mkdir(cached_thumbnails_path)
        if not os.path.exists(temp_images_path):
            os.mkdir(temp_images_path)
        self.video_folder_path = folder
        self.movies: dict[str, Movie] = {}
        self.current_movie = None
        self.scavenge()

    def scavenge(self):
        """
        Scavenge the movie folder for movies.
        """
        print("MU >> Scavenging movies at " + self.video_folder_path)
        exts = [".mkv", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm"]
        # get names of all movies in folder
        movies = os.listdir(self.video_folder_path)
        movies = [x for x in movies if os.path.splitext(x)[1] in exts]
        full_paths = [os.path.join(self.video_folder_path, x) for x in movies]
        print(f"MU >> Found {len(movies)} movies:")
        for movie, _ in zip(movies, full_paths):
            print("- " + movie)

        # update movies
        self.movies = {}
        for movie, full_path in zip(movies, full_paths):
            movie = os.path.splitext(movie)[0]
            self.movies[movie] = Movie(movie, full_path)
        return movies, full_paths

    def get_movies(self):
        """
        Get the movies.
        """
        return self.movies.values()

    def get_movie_names(self):
        """
        Get the movie names.
        """
        return self.movies.keys()

    def get_movie_by_name(self, name: str):
        """
        Get a movie by its name.
        """
        if name not in self.movies:
            return None
        return self.movies[name]

    def get_movie_by_path(self, path: str):
        """
        Get a movie by its url.
        """
        name = url_to_file_name(path)
        for movie in self.movies.values():
            if movie.name == name:
                print(f"MU >> Found movie at path {path} and name {name}")
                return movie
        print(f"MU >> Couldn't find movie at path {path}, name {name}")
        return None


user_path = os.path.expanduser("~")
video_folder_path = os.path.join(user_path, "Videos")
cached_thumbnails_path = r"frontend/public/cached_thubmnails"
relative_cached_thumbnails_path = r"cached_thubmnails"
temp_images_path = r"frontend/public/temp_images"
relative_temp_images_path = r"temp_images"
movies = Movies(video_folder_path)
