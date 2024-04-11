import threading
import time
from movie_utils import Movie
from vlc_python import get_audio_devices
import vlc


class VLC_Instance:
    def __init__(self) -> None:
        # kill all vlc instances running on windows
        self.players: list[vlc.MediaPlayer] = []
        self.instances: list[vlc.Instance] = []
        self.is_open_flag = False
        print("[VLC] VLC object created.")

    def open(self, lang_to_guid: dict[str, str], movie: Movie):
        for player in self.players:
            try:
                player.stop()
            except Exception as e:
                print("[VLC] Error stopping player", e)
        self.players = []
        for instance in self.instances:
            try:
                instance.release()
            except Exception as e:
                print("[VLC] Error releasing instance", e)
        self.instances = []

        print("[VLC] open()")
        self.is_open_flag = True

        ordered_tracks = [(index, lang) for (index, lang) in enumerate(movie.get_audio_tracks()) if lang in lang_to_guid]
        print("[VLC] ordered_tracks", ordered_tracks)
        if len(ordered_tracks) == 0:
            raise Exception("No tracks to play")

        devs = get_audio_devices()
        print("[VLC] Audio devices", devs.keys())

        for i, (index, lang) in enumerate(ordered_tracks):
            command1 = f'--audio-track={index} {"" if i != (len(ordered_tracks) - 1) else "--video-on-top"} --aout=waveout --waveout-audio-device="{lang_to_guid[lang][:31]} ($1,$64)"'
            print("[VLC]", command1)
            # subprocess.Popen(command1, cwd="C:\\Program Files\\VideoLAN\\VLC", shell=True)
            instance: vlc.Instance = vlc.Instance(command1)
            print("[VLC] Instance created")
            player: vlc.MediaPlayer = instance.media_player_new()
            print("[VLC] Player created")
            # use ouput device
            self.players.append(player)
            player.set_mrl(movie.path)
            print("[VLC] Set mrl to", movie.path)
            print("[VLC] Audio output device is", player.audio_output_device_get())
            print(f"[VLC] {lang_to_guid[lang]} in devs", lang_to_guid[lang] in devs)
            # if devs[lang_to_guid[lang]] is None:
            #    print("[VLC] Audio output device is None")
            # elif lang_to_guid[lang] not in devs:
            #    print(f"[VLC] {lang_to_guid[lang]} not in devices")
            # else:
            #    print("Everything is fine")
            # print("[VLC] Setting audio output device to", devs[lang_to_guid[lang]])
            # player.audio_output_device_set(None, devs[lang_to_guid[lang]])
            # print("[VLC] Set audio output device to", devs[lang_to_guid[lang]])
            # print bunch of information with vlc about the instance

            print(f"[VLC] Interface {index} will play track: {lang} with output device: {lang_to_guid[lang]}")

        self.play()
        player.set_fullscreen(True)
        # self.default_fullscreen()

        for player in self.players:
            player.set_time(10)
            # print audio device in use
            print(f"[VLC] Interface {player.audio_output_device_get()}")
            # output timestamp
            print(f"[VLC] Interface {player.get_time()}")
        # self.set_volume(30)

    def close(self):
        for player in self.players:
            player.stop()
        self.is_open_flag = False

    def is_open(self):
        return self.is_open_flag

    def pause(self):
        for instance in self.players:
            instance.pause()

    def default_fullscreen(self):
        self.players[0].set_fullscreen(True)

    def set_volume(self, volume):
        for i, http_hook in enumerate(self.http_hooks):
            http_hook.set_volume(volume)
            print(f"[VLC] Interface {i} volume set to {volume}.")

    def play(self):
        for instance in self.players:
            instance.play()

    def sync(self, to_0=False):
        attributes = self.http_hooks[0].get_attributes()
        timestamp = 0 if to_0 else int(attributes["time"]) + 2

        def asd(hook, timestamp, start):
            hook.goto(timestamp, attributes)
            # print elapsed time in milliseconds
            print(f"[VLC] Interface {hook.port - 8080} synced in {1000 * (time.time() - start)} milliseconds.")

        start = time.time()
        print(f"[VLC] Interface 0 timestamp: {timestamp}")
        threads = [threading.Thread(target=asd, args=(http_hook, timestamp, start)) for http_hook in self.http_hooks]

        for thread in threads:
            thread.start()

        # print(f"[VLC] Synced in {time.time() - start} seconds.")

    def move(self, seconds):
        attributes = self.http_hooks[0].get_attributes()
        timestamp = int(attributes["time"]) + seconds
        for i, http_hook in enumerate(self.http_hooks):
            http_hook.goto(timestamp, attributes)
            print(f"[VLC] Interface {i} moved by {seconds} seconds.")


inst = VLC_Instance()

"""Module to use VLC HTTP interface

@author Shubham Jain <shubham.jain.1@gmail.com>
@license MIT License

VLC provides an HTTP interface (by default disabled) at 127.0.0.1:8080.
I have written some basic functions to work on the interface.

Example:

vlc = vlc_http()

vlc.play_pause()
vlc.seek(5)  #Seek 5 seconds from current position
vlc.set_volume(100) #Set volume to 100%

"""
