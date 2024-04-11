// ignore_for_file: library_private_types_in_public_api, prefer_const_constructors, avoid_print, non_constant_identifier_names
import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

String gssUuid(String code) => '0000$code-0000-1000-8000-00805f9b34fb';

final CMD_SERVICE_UUID = gssUuid('181a');
final CMD_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E".toLowerCase();

class WifiCredentialsScreen extends StatefulWidget {
  final String macAddress;

  const WifiCredentialsScreen({required this.macAddress, Key? key})
      : super(key: key);

  @override
  _WifiCredentialsScreenState createState() => _WifiCredentialsScreenState();
}

class _WifiCredentialsScreenState extends State<WifiCredentialsScreen> {
  bool _passwordVisible = false;
  bool _isButtonDisabled = false;
  final _ssidController = TextEditingController();
  final _passwordController = TextEditingController();

  StreamSubscription? scanSubscription;
  StreamSubscription? stream;

  @override
  void initState() {
    super.initState();
    initBluetooth();
  }

  Future<void> initBluetooth() async {
    if (await FlutterBluePlus.isAvailable == false) {
      print("[BLE] Bluetooth not supported by this device");
      return;
    }

    if (Platform.isAndroid) {
      await FlutterBluePlus.turnOn();
      print("[BLE] Bluetooth turned on");
    }

    await FlutterBluePlus.adapterState
        .where((s) => s == BluetoothAdapterState.on)
        .first;
    print("[BLE] Bluetooth adapter is on");

    List<String> seen = [];

    scanSubscription = FlutterBluePlus.scanResults.listen((results) async {
      for (ScanResult r in results) {
        var macAddress = r.device.remoteId.toString();
        print("[BLE] Found device: $macAddress");
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
                stream = characteristic.value.listen((value) async {
                  final response = String.fromCharCodes(value);
                  // convert dataToSend to from list of integers to string
                  String dataToSendStr = String.fromCharCodes(command);
                  if (handleReceivedValue(response, dataToSendStr)) {
                    print("[BLE] Received expected data: $response");
                    stream?.cancel();
                    await r.device.disconnect();
                    Navigator.pop(context);
                  }
                });
                // Write to the characteristic
                await characteristic.write(command);
                await Future.delayed(Duration(seconds: 5), () async {
                  stream?.cancel();
                  if (mounted) {
                    await r.device.disconnect();
                    Navigator.pop(context);
                  }
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
        print("[BLE] Successfully sent calibration command");
        return true;
      }
      print("[BLE] Couldn't send calibration command");
      return false;
    }
    print("[BLE] $command != $commandToSend");
    return false;
  }

  @override
  void dispose() {
    scanSubscription?.cancel();
    stream?.cancel();
    FlutterBluePlus.stopScan();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Enter WiFi Credentials'),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: <Widget>[
            TextField(
              controller: _ssidController,
              decoration: InputDecoration(
                border: OutlineInputBorder(),
                labelText: 'Wifi SSID',
              ),
            ),
            SizedBox(height: 10),
            TextField(
              controller: _passwordController,
              obscureText: !_passwordVisible,
              decoration: InputDecoration(
                border: OutlineInputBorder(),
                labelText: 'Password',
                suffixIcon: IconButton(
                  icon: Icon(
                    _passwordVisible ? Icons.visibility : Icons.visibility_off,
                  ),
                  onPressed: () {
                    setState(() {
                      _passwordVisible = !_passwordVisible;
                    });
                  },
                ),
              ),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isButtonDisabled
                  ? null
                  : () async {
                      if (mounted) {
                        setState(() {
                          _isButtonDisabled = true;
                        });
                      }

                      bool setupResult = await wifiSetup(
                        _ssidController.text,
                        _passwordController.text,
                      );

                      print('Wifi setup result: $setupResult');
                      if (setupResult) {
                        Future.microtask(() => Navigator.of(context).pop());
                      } else {
                        if (mounted) {
                          setState(() {
                            _isButtonDisabled = false;
                          });
                        }
                      }
                    },
              child: Text('Save'),
            ),
          ],
        ),
      ),
    );
  }

}
