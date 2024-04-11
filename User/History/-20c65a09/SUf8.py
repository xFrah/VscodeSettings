import websocket

if __name__ == "__main__":
    ws = websocket.WebSocket()
    ws.connect("ws://127.0.0.1:3001")
    ws.send("Hello, World".encode("utf-8"))
    ws.close()
