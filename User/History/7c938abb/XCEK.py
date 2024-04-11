import sounddevice as sd

# list output devices
print(sd.query_devices(kind={'output'}), )

