# importing vlc module
import vlc

# creating vlc media player object
media = vlc.MediaPlayer("1.mp4")

# disable video, only play audio
media.video_set_visible(False)

# select the audio trackà
print(media.audio_output_device_enum())

# start playing video
media.play()
