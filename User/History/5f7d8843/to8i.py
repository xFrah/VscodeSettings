import json


def get_streams():
    with open("config.json", "r") as f:
        streams = json.load(f)
    return streams["streams"]

def get_padiglioni():
    ids = get_streams()