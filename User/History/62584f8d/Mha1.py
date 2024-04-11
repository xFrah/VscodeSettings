# importing vlc module
import time
import vlc

instance = vlc.Instance()
p = instance.media_player_new()

p.play()

def get_audio_devices(player):
    p = vlc.Instance()

device = instance.audio_output_device_enum()
while device:
    print("playing on...")
    print(device.contents.device)
    print(device.contents.description)

    p.audio_output_device_set(None, device.contents.device)
    time.sleep(3)

    device = device.contents.next

p.stop()
