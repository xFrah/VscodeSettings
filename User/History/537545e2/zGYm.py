import vlc


def play_audio_tracks(video_path, audio_outputs):
    # Create VLC instance
    vlc_instance = vlc.Instance()

    # Create media player
    player = vlc_instance.media_player_new()

    # Load the video file
    media = vlc_instance.media_new(video_path)
    player.set_media(media)

    # Start playing the video
    audio_outputs = vlc.libvlc_audio_output_device_list_get(player)
    player.play()

    # Set audio output for each track
    for i, audio_output in enumerate(audio_outputs):

        # Switch to the next audio track
        player.audio_set_track(i + 1)

        # Wait for a moment before switching to the next track
        vlc.libvlc_audio_output_device_list_release(audio_outputs)
        vlc.libvlc_audio_output_device_list_get(player)

    # Wait for the playback to finish
    duration = player.get_length()
    vlc.libvlc_media_player_play(player)
    while vlc.libvlc_media_player_get_state(player) != vlc.State.Ended:
        pass

    # Stop the player
    player.stop()

    # Release the player and media objects
    player.release()
    media.release()
    vlc_instance.release()


# Specify the path to the video file
video_path = "Maze Runner - La Fuga (2015) 2160p H265 10 bit ita eng AC3 5.1 sub ita eng Licdom.mkv"

# Specify the audio outputs (device names) for each track
audio_outputs = ["Speaker (Realtek(R) Audio)", "Altoparlanti (JBL Charge 5)"]

# Call the function to play audio tracks on different outputs
play_audio_tracks(video_path)
