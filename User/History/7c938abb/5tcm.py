import sounddevice as sd

devices = sd.query_devices()
output_devices = [device for device in devices if device['max_output_channels'] > 0]

for device in output_devices:
    print("Device ID:", device['name'])
    print("Device Name:", device['name'])
    print("Host API:", device['hostapi'])
    print("Max Output Channels:", device['max_output_channels'])
    print()