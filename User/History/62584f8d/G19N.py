# importing vlc module
import vlc

# creating vlc media player object
media = vlc.MediaPlayer("1.mp4")

# disable video, only play audio
media.video_set_visible(False)

# select the audio track
media.audio_set_track(1)



# start playing video
media.play()
