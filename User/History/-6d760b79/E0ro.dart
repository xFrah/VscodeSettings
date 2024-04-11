// ignore_for_file: library_private_types_in_public_api, prefer_const_constructors, avoid_print, non_constant_identifier_names
import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

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
  List<int>? lastKnownCommand;
  final String cmd_uuid = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E".toLowerCase();
  late StreamSubscription notify_stream;

  @override
  void initState() {
    super.initState();
    _passwordVisible = false;
    discoverDeviceAndServices();
  }

  Future<void> discoverDeviceAndServices() async {
    // Ensure Bluetooth is available and turned on
    if (await FlutterBluePlus.isAvailable == false) {
      print("[BLE] Bluetooth not supported by this device");
      return;
    }
    if (Platform.isAndroid) await FlutterBluePlus.turnOn();
    await FlutterBluePlus.adapterState
        .where((s) => s == BluetoothAdapterState.on)
        .first;

    StreamSubscription? scanSubscription =
        FlutterBluePlus.scanResults.listen((results) async {
      for (ScanResult r in results) {
        var macAddress = r.device.remoteId.toString();
        if (macAddress == widget.macAddress) {
          scanSubscription?.cancel();
          await r.device.connect();
          await r.device.requestMtu(250);
          List<BluetoothService> services = await r.device.discoverServices();
          // ... do something with the services
          return;
        }
      }
    });
    await FlutterBluePlus.stopScan();
    await FlutterBluePlus.startScan();
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

  @override
  void dispose() {
    // Clean up the controller when the widget is disposed.
    _ssidController.dispose();
    _passwordController.dispose();
    super.dispose();
    notify_stream.cancel(); // TODO might be null?
  }

  /// Starts the process of setting up the wifi credentials on the BLE device
  Future<bool> wifiSetup(String newSsid, String newPassword,
      {int timeout = 5000}) async {
    String commandToSend = "command:wifi_save+$newSsid+$newPassword";
    List<int> wifiSetupCmd = Uint8List.fromList(commandToSend.codeUnits);

    try {
      // write to characteristic with uuid cmd_uuid
      BluetoothCharacteristic cmdChar = get_cmd_char();
      await cmdChar.write(wifiSetupCmd);
    } catch (e) {
      print("[BLE] [BLE] Couldn't send the wifi setup command:");
      print(e);
      return false;
    }

    print("[BLE] [BLE] Sent command $commandToSend");

    var startTime = DateTime.now().millisecondsSinceEpoch;
    // check if the device has received the save command
    while ((lastKnownCommand == null) &&
        (DateTime.now().millisecondsSinceEpoch - startTime) < timeout) {
      // non-blocking sleep
      await Future.delayed(Duration(milliseconds: 100));
    }

    if (lastKnownCommand == null) {
      print("[BLE] [BLE] No response from device");
      return false;
    }

    var lastKnownCommandStr = String.fromCharCodes(lastKnownCommand!);
    var length = lastKnownCommandStr.length;

    if (lastKnownCommandStr[length - 2] != "=") {
      print("[BLE] [BLE] Bad response from device: $lastKnownCommandStr");
      return false;
    }

    var command = lastKnownCommandStr.substring(0, length - 2);
    var result = lastKnownCommandStr[length - 1];

    if (command == commandToSend) {
      if (result == "1") {
        print("[BLE] [BLE] Wifi setup command sent successfully");
        return true;
      }
      print("[BLE] [BLE] Wifi setup command failed");
      return false;
    }
    print("[BLE] [BLE] $command != $commandToSend");
    return false;
  }
}
