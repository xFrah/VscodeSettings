## Project setup

- Download micropython 1.20.1 stubs using [this guide](https://micropython-stubs.readthedocs.io/en/latest/20_using.html);
- Use [micropy](https://github.com/BradenM/micropy-cli) to setup vscode environment/project, use the correct stubs;
- Disable Pylance for the workspace because it is bugged apparently.


## ESP32 setup

- Download micropython firmware .bin from [here](https://micropython.org/download/esp32/);
- Flash the firmware with 
````bash
esptool.py --chip esp32 --port /dev/ttyUSB0 erase_flash
````
````bash
esptool.py --chip esp32 --port /dev/ttyUSB0 --baud 115200 write_flash -z 0x1000 firmware.bin
````
- Flash all the .py files to the esp32(you can do it using pymakr extension in vscode).


## Libraries

- mrequests: https://github.com/SpotlightKid/mrequests/tree/master
- mqtt: https://github.com/micropython/micropython-lib/tree/master/micropython/umqtt.simple
- pn532: https://github.com/Carglglz/NFC_PN532_SPI
- adxl345: https://github.com/fanday/adxl345_micropython/blob/master/adxl345.py


## TODO

- Use [ugit](https://github.com/turfptax/ugit).
- Create BLE commands for registering RFID tags.
- Create fallback for when db download fails.
- Compile micropython with sqlite and use it.
- Create format for mqtt messages.
- Deepsleep routine.
- Wake up at night time and check for db updates.
- UART communcation with RFID antenna.