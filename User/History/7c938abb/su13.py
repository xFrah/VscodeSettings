import sounddevice as sd

devices = sd.query_devices()
output_devices = [device["name"] for device in devices if device["max_output_channels"] > 0]

print(output_devices)