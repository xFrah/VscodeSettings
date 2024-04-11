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
        filename = filename.removesuffix(".txt")
        with open("padiglioni/" + filename, "r") as f:
            padiglione_dict[] = int(f.read())
            print(f"[INFO] Loaded {filename}")
    print(f"[INFO] Starting dictionary: {padiglione_dict}")