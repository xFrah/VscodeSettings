import json
import subprocess
import comtypes
from pycaw.pycaw import AudioUtilities, IMMDeviceEnumerator, EDataFlow, DEVICE_STATE
from pycaw.constants import CLSID_MMDeviceEnumerator


def get_interfaces_guids(State=DEVICE_STATE.ACTIVE.value):
    """
    Get a list of active audio devices.
    """

    try:
        devices = []
        Flow = EDataFlow.eRender.value  # 0

        deviceEnumerator = comtypes.CoCreateInstance(CLSID_MMDeviceEnumerator, IMMDeviceEnumerator, comtypes.CLSCTX_INPROC_SERVER)
        if deviceEnumerator is None:
            return devices

        collection = deviceEnumerator.EnumAudioEndpoints(Flow, State)
        if collection is None:
            return devices

        count = collection.GetCount()
        for i in range(count):
            dev = collection.Item(i)
            if dev is not None:
                if not ": None" in str(AudioUtilities.CreateDevice(dev)):
                    devices.append(AudioUtilities.CreateDevice(dev))

        for i, dev in enumerate(devices):
            s = dev.find("{", 1)
            e = dev.find("}", s)

            guid = dev[s + 1 : e]

            if s != -1 and e != -1:
                devices[i] = guid
        return devices
    except Exception as e:
        print(e)
        return []


class Interface:
    def __init__(self, guid: str) -> None:
        self.guid = guid
        self.name = guid


class Interfaces:
    def __init__(self) -> None:
        self.guids: list[str] = []
        try:
            self.guids = get_interfaces_guids()
        except Exception as e:
            print(e)

        self.interfaces: list[Interface] = [Interface(guid) for guid in self.guids]
        self.update_interface_names(self.load_interface_names())

    def get_audio_interfaces(self):
        """
        Get the audio interfaces.
        """
        return self.interfaces

    def save_interface_names(self, new_interface_names: dict[str, str]):
        """
        Save the translation between the interface name and the interface guid.
        """
        with open("interface_names.json", "w") as f:
            json.dump(new_interface_names, f)

    def update_interface_names(self, update_dict: dict[str, str], include_new_interfaces: bool = True):
        """
        Update the translation between the interface name and the interface guid.
        """
        # get the current translation from the file
        prev_interface_names = self.load_interface_names()
        # update the translation with the new interface_names
        for guid, name in update_dict.items():
            prev_interface_names[guid] = name
        if include_new_interfaces:
            for interface in self.interfaces:
                if interface.guid not in prev_interface_names:
                    prev_interface_names[interface.guid] = interface.name
        # save the updated translation
        self.save_interface_names(prev_interface_names)

    def load_interface_names(self):
        if not os.path.exists("interface_names.json"):
            return {}
        with open("interface_names.json", "r") as f:
            return json.load(f)


def get_track_name(stream: dict):
    """
    Get the name of the audio track from the stream info.
    """

    title = stream["tags"]["title"] if "tags" in stream and "title" in stream["tags"] else None
    lang = stream["tags"]["language"] if "tags" in stream and "language" in stream["tags"] else None
    name = ""
    if lang is not None:
        name += lang
    if title is not None:
        name += " - " + title
    if name == "":
        name = stream["index"]
    return name


def get_audio_tracks_info(movie_file):
    """
    Get the audio tracks of a movie file.
    """

    command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "a", movie_file]
    # try:
    output = subprocess.check_output(command)
    # except subprocess.CalledProcessError as e:
    #    return print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    info = json.loads(output)
    audio_tracks = []

    # sort info["streams"] by index
    info["streams"] = sorted(info["streams"], key=lambda k: k["index"])

    for stream in info["streams"]:
        if stream["codec_type"] == "audio":
            audio_track = {
                "name": get_track_name(stream),
                "index": stream["index"],
            }
            audio_tracks.append(audio_track)

    return [x["name"] for x in audio_tracks]


interfaces = Interfaces()
