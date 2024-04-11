# importing vlc module
import time
import vlc


def get_audio_devices():
    devices = []
    p = vlc.Instance().media_player_new()

    device = p.audio_output_device_enum()
    while device:
        # p.audio_output_device_set(None, device.contents.device)
        devices.append((device.contents.device, device.contents.description))
        device = device.contents.next

    return devices


print(get_audio_devices())
