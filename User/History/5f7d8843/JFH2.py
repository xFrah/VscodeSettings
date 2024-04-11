import json
from typing import Union, Dict, Any
import os


def get_padiglioni_config() -> Dict[str, Dict[str, Any]]:
    """
    Returns the streams as a dictionary of shape dict[padiglione_id: str, rtsp_link: str].
    """
    with open("config.json", "r") as f:
        streams = json.load(f)
    return streams


def get_padiglione_dict() -> dict[str, int]:
    """
    Returns a dictionary with the padiglioni as keys and the number of people as values.
    """
    ids = get_padiglioni_config().keys()
    padiglione_dict = {}
    padiglioni_folder = os.listdir("padiglioni")
    for filename in ids:
        if filename in padiglioni_folder:
            with open("padiglioni/" + filename, "r") as f:
                n = f.read()
                padiglione_dict[filename.removesuffix(".txt")] = int(n)
                print(f"[INFO] Caricato padiglione {filename} con {n} persone.")
        else:
            padiglione_dict[filename] = 0
            print(f"[INFO] Creato padiglione {filename}.")
    return padiglione_dict


def save_padiglioni(padiglione_dict: Dict[str, int]) -> None:
    """
    Saves the padiglioni dictionary to the disk.
    """
    for key in padiglione_dict:
        buf = padiglione_dict[key]
        with open("padiglioni/" + key + ".txt", "w") as f:
            f.write(str(buf))
        print(f"[INFO] Saved {key}:{buf}")
    print("[INFO] Conteggio padiglioni salvato con successo.")


def ensure_padiglioni_dir() -> None:
    """
    Creates the padiglioni directory if it doesn't exist.
    """
    if not os.path.exists("padiglioni"):
        os.mkdir("padiglioni")
        print("[INFO] Creata cartella padiglioni.")


ensure_padiglioni_dir()
