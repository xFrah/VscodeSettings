# make python script to test the broker


if __name__ == "__main__":
    client = mqtt.Client()
    client.connect("localhost", 1883, 60)