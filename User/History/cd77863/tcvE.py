import paho.mqtt.client as mqtt

mqtt_host = "homeassistant.local"
port = 8080
lock = RLock()
padiglione_dict = {}


def update_handler():
    client = mqtt.Client(client_id="mqtt_user", clean_session=True, userdata=None, protocol=mqtt.MQTTv311, transport="tcp")
    client.username_pw_set("mqtt_user", "Beam2020")
    try:
        client.connect(mqtt_host, port=1883, keepalive=100)
    except:
        print("[FATAL] Couldn't open mqtt connection, RESTART THE SCRIPT")
        return
    print("[INFO] Connected to mqtt")
    last_update = datetime.datetime.now()
    last_save = datetime.datetime.now()
    delta = datetime.timedelta(seconds=5)
    save_delta = datetime.timedelta(seconds=60)
    while True:
        time.sleep(0.1)
        now = datetime.datetime.now()
        if now - last_update > delta:
            for key in padiglione_dict:
                buf = padiglione_dict[key]
                client.publish(f"pad{key}", buf)
                print(f"[LOG] Sent {key}:{buf}")
            last_update = datetime.datetime.now()
        if now - last_save > save_delta:
            print("[INFO] Saving, DON'T CLOSE THE SCRIPT")
            for key in padiglione_dict:
                buf = padiglione_dict[key]
                with open("padiglioni/" + key + ".txt", "w") as f:
                    f.write(str(buf))
                print(f"[INFO] Saved {key}:{buf}")
            last_save = datetime.datetime.now()
            print("[INFO] Finished saving.")
