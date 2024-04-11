# importing vlc module
import time
import vlc

# creating vlc media player object
p = vlc.MediaPlayer("Maze Runner - La Fuga (2015) 2160p H265 10 bit ita eng AC3 5.1 sub ita eng Licdom.mkv")

p.play()

# set minimized
# p.video_set_mouse_input(False)


# set fullscreen
# p.toggle_fullscreen()


device = p.audio_output_device_enum()
while device:
    print("playing on...")
    print(device.contents.device)
    print(device.contents.description)

    p.audio_output_device_set(None, device.contents.device)
    time.sleep(3)

    device = device.contents.next

p.stop()
