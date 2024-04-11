import datetime
import os
import socket
from _thread import *
import time
import paho.mqtt.client as mqtt

host = socket.gethostname()
mqtt_host = "homeassistant.local"
port = 8080
ThreadCount = 0
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


def client_handler(connection):
    # make split counts parsing the string <id_padiglione>:<increment>
    # connection.send(str.encode('You are now connected to the replay server...'))
    while True:
        try:
            data = connection.recv(2048)
        except:
            print("[INFO] A client disconnected, shutting thread down")
            return
        message = data.decode("utf-8")
        split_msg = message.split(":")
        try:
            padiglione, inc = split_msg[0], int(split_msg[1])
        except IndexError:
            print("[ERROR] Received corrupted or empty packet from a client.")
            time.sleep(2)
            continue
        with lock:
            if padiglione not in padiglione_dict:
                print(f"[INFO] Found new padiglione: {padiglione}")
                padiglione_dict[padiglione] = 0
            padiglione_dict[padiglione] += inc
            # print(f"[INFO] PAD-{padiglione} incremented by {inc}: {padiglione_dict[padiglione]}")


def accept_connections(ServerSocket):
    Client, address = ServerSocket.accept()
    print("[INFO] Connected to: " + address[0] + ":" + str(address[1]))
    start_new_thread(client_handler, (Client,))


def start_server(host, port):
    global padiglione_dict
    padiglione_dict = {}
    for filename in os.listdir("padiglioni"):
        with open("padiglioni/" + filename, "r") as f:
            padiglione_dict[filename.removesuffix(".txt")] = int(f.read())
            print(f"[INFO] Loaded {filename}")
    print(f"[INFO] Starting dictionary: {padiglione_dict}")
    start_new_thread(update_handler, ())
    ServerSocket = socket.socket()
    try:
        ServerSocket.bind((host, port))
    except socket.error as e:
        print(str(e))
    print(f"[INFO] Server is listing on the port {port}...")
    ServerSocket.listen()
    while True:
        accept_connections(ServerSocket)


start_server(host, port)
