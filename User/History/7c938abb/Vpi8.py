import sounddevice as sd

# list output devices
sd.query_devices(kind="output")

print(len(sd.query_devices(kind="output")))