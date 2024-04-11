# importing vlc module
import vlc

# creating vlc media player object
media = vlc.MediaPlayer("Maze Runner - La Fuga (2015) 2160p H265 10 bit ita eng AC3 5.1 sub ita eng Licdom.mkv")

# select the audio trackà
print(media.audio_output_device_get())

# start playing video
media.play()
