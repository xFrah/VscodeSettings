import json


def get_rtsp_links():
    """
    Returns the streams as a dictionary of shape dict[padiglione_id: , rtsp_link].
    """
    with open("config.json", "r") as f:
        streams = json.load(f)
    return streams["padiglioni"]


def get_padiglioni() -> dict[str, int]:
    """
    Returns a dictionary with the padiglioni as keys and the number of people as values.
    """
    ids = get_rtsp_links().keys()
    padiglione_dict = {}
    padiglioni_folder = os.listdir("padiglioni")
    for filename in ids:
        if filename in padiglioni_folder:
            with open("padiglioni/" + filename, "r") as f:
                padiglione_dict[filename.removesuffix(".txt")] = int(f.read())
                print(f"[INFO] Loaded {filename}")
        else:
            padiglione_dict[filename] = 0
            print(f"[INFO] Created {filename}")
    print(f"[INFO] Starting dictionary: {padiglione_dict}")


def save_padiglioni(padiglione_dict: dict[str, int]) -> None:
    """
    Saves the padiglioni dictionary to the disk.
    """
    for key in padiglione_dict:
        buf = padiglione_dict[key]
        with open("padiglioni/" + key + ".txt", "w") as f:
            f.write(str(buf))
        print(f"[INFO] Saved {key}:{buf}")
    print("[INFO] Finished saving.")


def ensure_padiglioni_dir() -> None:
    """
    Creates the padiglioni directory if it doesn't exist.
    """
    if not os.path.exists("padiglioni"):
        os.mkdir("padiglioni")
        print("[INFO] Created padiglioni directory")
