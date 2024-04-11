import sounddevice as sd

devices = sd.query_devices()
output_devices = [device for device in devices if device["max_output_channels"] > 0 and device["hostapi"] == 0]

print(output_devices)