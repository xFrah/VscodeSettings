import json


def get_streams():
    with open("config.json", "r") as f:
        streams = json.load(f)
    return streams["streams"]
