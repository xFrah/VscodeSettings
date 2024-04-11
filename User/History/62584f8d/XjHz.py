# importing vlc module
import time
import vlc


def get_audio_devices():
    devices = []
    p = vlc.Instance().media_player_new()

    device = p.audio_output_device_enum()
    while device:
        print("playing on...")
        print(device.contents.device)
        print(device.contents.description)

        # p.audio_output_device_set(None, device.contents.device)
        print(f"Audio device set to {device.contents.description}")
        time.sleep(3)

        device = device.contents.next
