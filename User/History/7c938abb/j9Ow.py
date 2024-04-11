import sounddevice as sd

# list output devices
asd = sd.query_devices(kind="output")

# check if type is dict or DeviceList

