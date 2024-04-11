# importing vlc module
import vlc

# creating vlc media player object
media = vlc.MediaPlayer("1.mp4")

# disable video, only play audio
media.video_set_visible(False)

# select the audio trackà
media.

# start playing video
media.play()
