import json


def get_streams():
    with open("config.json", "r") as f:
        streams = json.load(f)
    return streams["streams"]


def get_padiglioni():
    ids = get_streams().keys()
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

def save_padiglioni(padiglione_dict: dict):
    