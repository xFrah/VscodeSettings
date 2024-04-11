class mqtt_client:
    def __init__(self) -> None:
        client = mqtt.Client()
        client.connect("localhost", 1883, 60)
        # on_message
        client.subscribe("kit/#")
        client.on_message = on_message
        client.loop_start()
