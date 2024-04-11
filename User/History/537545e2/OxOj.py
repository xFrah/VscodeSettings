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
available_outputs = vlc.Instance().audio_output_device_enum()

# Retrieve audio tracks
audio_tracks = media_player.audio_get_track_description()

# Print audio track information
print("Audio Tracks:")
for track in audio_tracks:
    print(" - Track {}: {}".format(track[0], track[1].decode("utf-8")))


def get_available_audio_outputs():
    instance = vlc.Instance()
    audio_output_list = instance.audio_output_device_list_get()

    # Get the device names
    device_names = []
    for device in audio_output_list:
        device_name = device.decode("utf-8")
        device_names.append(device_name)

    # Release the device list
    instance.audio_output_device_list_release(audio_output_list)

    return device_names


# Call the function to get the available audio output device names
output_devices = get_available_audio_outputs()

# Print the device names
for device in output_devices:
    print(device)
