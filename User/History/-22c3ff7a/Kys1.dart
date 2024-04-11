import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'dart:io';

String gssUuid(String code) => '0000$code-0000-1000-8000-00805f9b34fb';

final CMD_SERVICE_UUID = gssUuid('181a');
final CMD_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E".toLowerCase();

class BLECalibration extends StatefulWidget {
  final String macAddress;
  BLECalibration({
    Key? key,
    required this.macAddress,
  }) : super(key: key);

  @override
  _BLECalibrationState createState() => _BLECalibrationState();
}

class _BLECalibrationState extends State<BLECalibration> {
  final targetCharacteristicUuid =
      "YOUR_TARGET_UUID"; // Replace with your target Characteristic UUID
  // command to send to device is a string that needs to be converted to a list of integers
  final command = "command:magnetometer_calibrate".codeUnits;

  StreamSubscription? scanSubscription;
  StreamSubscription? streamsub2;

  @override
  void initState() {
    super.initState();
    initBluetooth();
  }

  Future<void> initBluetooth() async {
    if (await FlutterBluePlus.isAvailable == false) {
      print("Bluetooth not supported by this device");
      return;
    }

    if (Platform.isAndroid) {
      await FlutterBluePlus.turnOn();
    }

    await FlutterBluePlus.adapterState
        .where((s) => s == BluetoothAdapterState.on)
        .first;

    List<String> seen = [];

    scanSubscription = FlutterBluePlus.scanResults.listen((results) async {
      for (ScanResult r in results) {
        var macAddress = r.device.remoteId.toString();
        if (seen.contains(macAddress)) continue;
        seen.add(macAddress);
        if (macAddress == widget.macAddress) {
          print("[BLE] Found device: ${r.device.localName}!");
          scanSubscription?.cancel(); // Stop scanning once we found our device
          await r.device.connect();
          await r.device.requestMtu(250);
          List<BluetoothService> services = await r.device.discoverServices();
          for (var service in services) {
            for (var characteristic in service.characteristics) {
              if (characteristic.uuid.toString() == CMD_RX_CHAR_UUID) {
                print("[BLE] Found characteristic: ${characteristic.uuid}!");
                await characteristic.setNotifyValue(true);
                StreamSubscription? stream;
                int res = 0;
                stream = characteristic.value.listen((value) {
                  final response = String.fromCharCodes(value);
                  // convert dataToSend to from list of integers to string
                  String dataToSendStr = String.fromCharCodes(command);
                  if (!handleReceivedValue(response, dataToSendStr)) {
                    print("[BLE] Received unexpected data: $response");
                  } else {
                    print("[BLE] Received expected data: $response");
                  }
                  stream?.cancel();
                });
                // Write to the characteristic
                await characteristic.write(command);
                await Future.delayed(Duration(seconds: 1), () {
                  Navigator.pop(context);
                });
                return;
              } else {
                print("[BLE] Found characteristic: ${characteristic.uuid}...");
              }
            }
          }
        } else {
          print(
              "[BLE] ${r.device.remoteId.toString()} != ${widget.macAddress}");
        }
      }
    });

    await FlutterBluePlus.stopScan();
    await FlutterBluePlus.startScan();
  }

  bool handleReceivedValue(String lastKnownCommandStr, String commandToSend) {
    int length = lastKnownCommandStr.length;

    if (length < 2) return false;

    if (lastKnownCommandStr[length - 2] != "=") {
      print("[BLE] Bad response from device: $lastKnownCommandStr");
      return false;
    }

    var command = lastKnownCommandStr.substring(0, length - 2);
    var result = lastKnownCommandStr[length - 1];

    if (command == commandToSend) {
      if (result == "1") {
        print("[BLE] Wifi setup command sent successfully");
        return true;
      }
      print("[BLE] Wifi setup command failed");
      return false;
    }
    print("[BLE] $command != $commandToSend");
    return false;
  }

  @override
  void dispose() {
    scanSubscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Bluetooth Search & Connect")),
      body: Center(child: Text("Searching for Bluetooth devices...")),
    );
  }
}
