# importing vlc module
import vlc


def get_audio_devices():
    devices = dict()
    instance = vlc.Instance()
    p = instance.media_player_new()

    device = p.audio_output_device_enum()
    while device:
        # p.audio_output_device_set(None, device.contents.device)
        if device.contents.device != b"":
            devices[device.contents.description.decode("utf-8")] = device.contents.device
        device = device.contents.next

    instance.release()

    return devices


print(get_audio_devices())
