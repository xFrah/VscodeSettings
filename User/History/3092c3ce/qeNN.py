import json
import os
import time
import streamlit as st
import streamlit.components.v1 as components
import audio_utils
import movie_utils
import vlc_interface
import urllib.parse

# change settings menu to have just a button that changes state to interface_name_changer
st.set_page_config(page_title="Movie Player", page_icon="🎬")
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
# set light mode
st.markdown(
    """
    <style>
    body {
        color: #111;
        background-color: #eee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def url_to_file_name(url):
    parsed_url = urllib.parse.urlparse(url)
    file_name = urllib.parse.unquote(parsed_url.path)  # Decode URL
    file_name = file_name.split("/")[-1]  # Extract the file name from the path
    return file_name


def film_choice():
    # write a centered text to the app
    st.title("Select a movie...")
    imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")

    imageUrls = []
    print("[WEBAPP] Showing {} movies".format(len(movie_utils.movies.get_movies())))
    for movie in movie_utils.movies.get_movies():
        if movie.cached_thumbnail is not None:
            # append only filename to the list
            imageUrls.append(movie.cached_thumbnail)
            print("[WEBAPP] Found cached thumbnail for movie {}".format(movie.name))
        else:
            # create placeholder image 400x400 with opencv and a label with the movie name
            pass

    print(imageUrls)

    selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)

    if selectedImageUrl is not None:
        movie = movie_utils.movies.get_movie_by_path(selectedImageUrl)
        print("[WEBAPP] Selected movie {}".format(movie.name))
        update_state({"state": "select_audio", "selected_movie": movie.name}, rerun=True)


def controls(status: dict[str, str]):
    current_movie = movie_utils.movies.get_movie_by_name(status["selected_movie"])
    if current_movie is None:
        st.write("Movie not found, restarting...")
        return update_state({"state": "home"}, rerun=True, delay=5)
    # get config from current_movie
    lang_to_guid = current_movie.get_config()
    if not vlc_interface.inst.is_open():
        try:
            vlc_interface.inst.open(lang_to_guid, current_movie)
        except Exception as i:
            print(i)
            print("[WEBAPP] Could not open VLC, restarting...")
            return update_state({"state": "home"}, rerun=True, delay=5)
        print("[WEBAPP] VLC opened")
    else:
        print("[WEBAPP] VLC already opened")
    # st.title(current_movie.name)
    st.markdown(f"<h1 style='text-align: center;'>🎬 {current_movie.name} 🎬‎‎ ‎ ‎ </h1><br>", unsafe_allow_html=True)
    # crate a row of 4 buttons to control the chosen movie
    _, col1, col2, col3, col4, col5, _ = st.columns([2, 1, 1, 1, 1, 1, 2])
    with col1:
        if st.button("⏮️"):
            vlc_interface.inst.move(-10)
            vlc_interface.inst.sync()
            st.write("Rewinded 10 seconds")
    with col2:
        if st.button("⏹️"):
            print("[WEBAPP] Stopping movie")
            vlc_interface.inst.close()
            update_state({"state": "home"}, rerun=True)
    with col3:
        if st.button("⏸️"):
            vlc_interface.inst.pause()
            st.write("Movie paused")
    with col4:
        if st.button("⏯️"):
            vlc_interface.inst.play()
            st.write("Playing...")
    with col5:
        if st.button("⏭️"):
            vlc_interface.inst.move(+10)
            vlc_interface.inst.sync()
            st.write("Skipped 10 seconds")

    def change_volume():
        vlc_interface.inst.set_volume(st.session_state.volume_slider)

    st.slider("Volume", 0, 200, 30, key="volume_slider", on_change=change_volume)


def get_guid_from_selectbox(selectbox: str):
    # get the guid from the selectbox, its the only thing in between brackets
    # get all text before last bracket
    # find last " (" in string

    last = selectbox.rfind(" - [")
    if last == -1:
        return selectbox
    res = selectbox[:last]
    print(f"[WEBAPP] Got name {res} from string {selectbox}")
    return selectbox[:last]


def interface_name_changer(status: dict[str, str]):
    """
    Change the names of the interfaces.
    """

    st.title("Change the names of the interfaces...")
    st.text_input("New name", key="new_name")
    interfaces = audio_utils.interfaces.get_audio_interfaces()
    st.selectbox(
        "Select audio interface",
        [x.guid if x.guid == x.name else f"{x.guid} - [{x.name}]" for x in interfaces],
        key="selectbox",
    )

    # place save and cancel button in a row
    _, col1, col2, _ = st.columns([5, 2, 2, 5])

    with col1:
        if st.button("Save"):
            # get guid from selectbox
            guid = get_guid_from_selectbox(st.session_state["selectbox"])
            # get new name from input
            new_name = st.session_state["new_name"]
            print("[WEBAPP] Renaming {} to {}".format(guid, new_name))
            audio_utils.interfaces.rename_interface(guid, new_name)
            st.experimental_rerun()

    with col2:
        if st.button("Back"):
            update_state({"state": "select_audio", "selected_movie": status["selected_movie"]}, rerun=True)


def update_state(state_dict: dict[str, str], rerun=False, delay=0):
    """
    Update the state of the software by writing to the json status file.
    """
    with open("status.json", "w") as f:
        json.dump(state_dict, f)

    print("[WEBAPP] Updated state to {}".format(state_dict))
    if rerun:
        time.sleep(delay)
        st.experimental_rerun()


def select_audio(status: dict[str, str]):
    """
    Select the audio tracks for the movie.
    """
    current_movie = movie_utils.movies.get_movie_by_name(status["selected_movie"])
    if current_movie is None:
        st.write("Movie not found")
        update_state({"state": "home"}, rerun=True, delay=5)

    st.title("Select the interface for each language...")
    interfaces = audio_utils.interfaces.get_audio_interfaces()
    config = current_movie.get_config()
    print("[WEBAPP] Found {} audio interfaces".format(len(interfaces)))
    # get audio tracks
    tracks = current_movie.get_audio_tracks()
    # for each track, get index of interface in interfaces which has the same guid as the one in the config
    # if no interface is found, set index to 0
    track_to_indeces = {}
    for lang in tracks:
        if lang not in config:
            index = 0
        else:
            guid = config[lang]
            index = audio_utils.interfaces.get_interface_index_by_guid(guid)
            if index is None:
                index = 0
            else:
                index += 1
        track_to_indeces[lang] = index

    for lang in tracks:
        st.selectbox(
            lang,
            ["No interface"] + [x.guid if x.guid == x.name else f"{x.guid} - [{x.name}]" for x in interfaces],
            key=lang,
            index=track_to_indeces[lang],
        )

    # place save and cancel button in a row
    _, col1, col2, _ = st.columns([5, 2, 2, 5])

    with col1:
        if st.button("Play"):
            track_to_guid = {
                track: get_guid_from_selectbox(st.session_state[track]) for track in tracks if st.session_state[track] != "No interface"
            }
            current_movie.save_config(track_to_guid)
            update_state({"state": "controls", "selected_movie": current_movie.name}, rerun=True)

    with col2:
        if st.button("Back"):
            update_state({"state": "home"}, rerun=True)

    _, col, _ = st.columns([1, 1, 1])
    with col:
        if st.button("Change interface names"):
            update_state({"state": "interface_name_changer", "selected_movie": current_movie.name}, rerun=True)


def get_status():
    """
    Get the status of the software by reading the json status file.
    """
    # create status file with state home if it does not exist
    if not os.path.exists("status.json"):
        default = {"state": "home"}
        with open("status.json", "w") as f:
            json.dump(default, f)
        return default
    # get status dict from json file
    with open("status.json", "r") as f:
        return json.load(f)


def main():
    status = get_status()
    print("[WEBAPP] Current state: {}".format(status))
    if status["state"] == "home":
        film_choice()
    elif status["state"] == "interface_name_changer":
        interface_name_changer(status)
    elif status["state"] == "select_audio":
        select_audio(status)
    elif status["state"] == "controls":
        controls(status)
    else:
        raise ValueError("Unknown state: {}".format(status["state"]))


if __name__ == "__main__":
    main()
