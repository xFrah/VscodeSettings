# importing vlc module
import vlc

# importing time module
import time


# creating vlc media player object
media_player = vlc.MediaPlayer()

# media object
media = vlc.Media("Maze Runner - La Fuga (2015) 2160p H265 10 bit ita eng AC3 5.1 sub ita eng Licdom.mkv")

# setting media to the media player
media_player.set_media(media)

# setting video scale
media_player.video_set_scale(0.6)

# start playing video
# media_player.play()

# wait so the video can be played for 5 seconds
# irrespective for length of video
# time.sleep(5)

# getting track
value: list[vlc.AudioOutputDevice] = media_player.audio_output_device_enum()

# Retrieve audio tracks
audio_tracks = media_player.audio_get_track_description()

# Print audio track information
print("Audio Tracks:")
for track in audio_tracks:
    print(" - Track {}: {}".format(track[0], track[1].decode("utf-8")))


# printing value
print("Audio Output Devices: ")
print(value)
device_names = [device for device in value]
