// ignore_for_file: library_private_types_in_public_api, prefer_const_constructors, avoid_print, non_constant_identifier_names
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

class WifiCredentialsScreen extends StatefulWidget {
  final BluetoothDevice device;
  final List<BluetoothService> services;

  const WifiCredentialsScreen(
      {required this.device, required this.services, Key? key})
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

  @override
  void initState() {
    super.initState();
    _passwordVisible = false;
    setupCmdCharacteristic();
  }

  BluetoothCharacteristic get_cmd_char() {
    for (BluetoothService service in widget.services) {
      for (BluetoothCharacteristic char in service.characteristics) {
        if (char.uuid.toString() == cmd_uuid) {
          return char;
        } else {
          print("[BLE] $cmd_uuid != ${char.uuid.toString()}");
        }
      }
    }
    throw Exception("Characteristic not found");
  }

  Future<void> setupCmdCharacteristic() async {
    for (BluetoothService service in widget.services) {
      for (BluetoothCharacteristic char in service.characteristics) {
        if (char.uuid.toString() == cmd_uuid) {
          await char.setNotifyValue(true);
          char.value.listen((value) {
            setState(() {
              lastKnownCommand = value;
            });
          });
          return;
        } else {
          print("[BLE] $cmd_uuid != ${char.uuid.toString()}");
        }
      }
    }
    throw Exception("Characteristic not found");
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
  }

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
