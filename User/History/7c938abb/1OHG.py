import sounddevice as sd

# list output devices
asd = sd.query_devices()

# check if type is dict
if type(asd) == dict:
    print("Dict")
elif asd is None:
    print("None")
else:
    for device in asd:
        print(device)

print(asd)
