# importing vlc module
import time
import vlc

# creating vlc media player object
media = vlc.MediaPlayer("Maze Runner - La Fuga (2015) 2160p H265 10 bit ita eng AC3 5.1 sub ita eng Licdom.mkv")

p.play()

device = p.audio_output_device_enum()
while device:
    print("playing on...")
    print(device.contents.device)
    print(device.contents.description)

    p.audio_output_device_set(None, device.contents.device)
    time.sleep(3)

    device = device.contents.next

p.stop()
