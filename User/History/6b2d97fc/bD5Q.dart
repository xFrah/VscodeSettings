import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:convert/convert.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:quick_blue/quick_blue.dart';

String gssUuid(String code) => '0000$code-0000-1000-8000-00805f9b34fb';

final CMD_SERVICE_UUID = gssUuid('181a');
final CMD_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E".toLowerCase();

var last_known_command = null;
bool is_connected = false;
bool ongoing_wifi_setup = false;

final GSS_SERV__BATTERY = gssUuid('180f');
final GSS_CHAR__BATTERY_LEVEL = gssUuid('2a19');

const WOODEMI_SUFFIX = 'ba5e-f4ee-5ca1-eb1e5e4b1ce0';

const WOODEMI_SERV__COMMAND = '57444d01-$WOODEMI_SUFFIX';
const WOODEMI_CHAR__COMMAND_REQUEST = '57444e02-$WOODEMI_SUFFIX';
const WOODEMI_CHAR__COMMAND_RESPONSE = WOODEMI_CHAR__COMMAND_REQUEST;

const WOODEMI_MTU_WUART = 247;

class PeripheralDetailPage extends StatefulWidget {
  final String deviceId;

  PeripheralDetailPage(this.deviceId);

  @override
  State<StatefulWidget> createState() {
    return _PeripheralDetailPageState();
  }
}

Future<bool> ensure_connection(String deviceId) async {
  // if the status is not connected, try to connect
  if (!is_connected) {
    QuickBlue.connect(deviceId);
    await Future.delayed(Duration(milliseconds: 500));
    // discover
    QuickBlue.discoverServices(deviceId);
    await Future.delayed(Duration(milliseconds: 500));
    return true;
  }
  return false;
}

Future<bool> wifi_setup(String deviceId, String new_ssid, String new_pass,
    {int timeout = 5000}) async {
  last_known_command = null;

  String command_to_send = "command:wifi_save+$new_ssid+$new_pass";

  Uint8List wifi_setup_cmd = Uint8List.fromList(command_to_send.codeUnits);

  try {
    QuickBlue.writeValue(deviceId, CMD_SERVICE_UUID, CMD_RX_CHAR_UUID,
        wifi_setup_cmd, BleOutputProperty.withResponse);
  } catch (e) {
    print("[BLE] Couldn't send the wifi setup command:");
    print(e);
  }

  // translate back cmd to string
  String cmd_str = String.fromCharCodes(wifi_setup_cmd);
  print("[BLE] Sent command $cmd_str");

  var start_time = DateTime.now().millisecondsSinceEpoch;
  // check if the device has received the save command
  while ((last_known_command == null) &&
      (DateTime.now().millisecondsSinceEpoch - start_time) < timeout) {
    // non blocking sleep
    await Future.delayed(Duration(milliseconds: 100));
  }

  if (last_known_command == null) {
    print("[BLE] No response from device");
    return false;
  }

  // at the end of last_known_commands, there is either =1 or =0
  var last_known_command_str = String.fromCharCodes(last_known_command);
  var length = last_known_command_str.length;

  // convert str = to byte
  if (last_known_command_str[length - 2] != "=") {
    print("[BLE] Bad response from device: $last_known_command_str");
    return false;
  }

  var command = last_known_command_str.substring(0, length - 2);
  var result = last_known_command_str[length - 1];

  if (command == command_to_send) {
    if (result == "1") {
      print("[BLE] Wifi setup command sent successfully");
      return true;
    }
    print("[BLE] Wifi setup command failed");
    return true;
  }
  print("[BLE] $command != $command_to_send");
  return false;
}

class _PeripheralDetailPageState extends State<PeripheralDetailPage> {
  @override
  void initState() {
    super.initState();
    QuickBlue.setConnectionHandler(_handleConnectionChange);
    QuickBlue.setServiceHandler(_handleServiceDiscovery);
    QuickBlue.setValueHandler(_handleValueChange);
  }

  @override
  void dispose() {
    super.dispose();
    QuickBlue.setValueHandler(null);
    QuickBlue.setServiceHandler(null);
    QuickBlue.setConnectionHandler(null);
  }

  void _handleConnectionChange(String deviceId, BlueConnectionState state) {
    print('_handleConnectionChange $deviceId, $state');
    if (state == BlueConnectionState.connected) {
      QuickBlue.discoverServices(deviceId);
      is_connected = true;
      print("[BLE] Connected to device");
    } else if (state == BlueConnectionState.disconnected) {
      is_connected = false;
      print("[BLE] Disconnected from device");
    }
  }

  void _handleServiceDiscovery(
      String deviceId, String serviceId, List<String> characteristicIds) {
    print('_handleServiceDiscovery $deviceId, $serviceId, $characteristicIds');
    for (var characteristicId in characteristicIds) {
      // try except
      try {
        if (characteristicId ==
            "6E400002-B5A3-F393-E0A9-E50E24DCCA9E".toLowerCase()) {
          QuickBlue.setNotifiable(deviceId, serviceId, characteristicId,
              BleInputProperty.indication);
          print("[BLE] Set cmd notifiable");
        } else {
          print(
              "[BLE] $characteristicId != 6E400002-B5A3-F393-E0A9-E50E24DCCA9E");
        }
      } catch (e) {
        print("[BLE] ERROOOOOOOR");
        print(e);
      }
    }
  }

  void _handleValueChange(
      String deviceId, String characteristicId, Uint8List value) {
    print(
        '_handleValueChange $deviceId, $characteristicId, ${hex.encode(value)}');
    if (characteristicId == CMD_RX_CHAR_UUID) {
      last_known_command = value;
    }
  }

  final serviceUUID = TextEditingController(text: WOODEMI_SERV__COMMAND);
  final characteristicUUID =
      TextEditingController(text: WOODEMI_CHAR__COMMAND_REQUEST);
  final binaryCode = TextEditingController(
      text: hex.encode([0x01, 0x0A, 0x00, 0x00, 0x00, 0x01]));

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('PeripheralDetailPage'),
      ),
      body: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: <Widget>[
              ElevatedButton(
                child: Text('connect'),
                onPressed: () {
                  QuickBlue.connect(widget.deviceId);
                },
              ),
              ElevatedButton(
                child: Text('disconnect'),
                onPressed: () {
                  QuickBlue.disconnect(widget.deviceId);
                },
              ),
            ],
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: <Widget>[
              ElevatedButton(
                child: Text('discoverServices'),
                onPressed: () {
                  QuickBlue.discoverServices(widget.deviceId);
                },
              ),
            ],
          ),
          TextField(
            controller: serviceUUID,
            decoration: InputDecoration(
              labelText: 'ServiceUUID',
            ),
          ),
          TextField(
            controller: characteristicUUID,
            decoration: InputDecoration(
              labelText: 'CharacteristicUUID',
            ),
          ),
          TextField(
            controller: binaryCode,
            decoration: InputDecoration(
              labelText: 'Binary code',
            ),
          ),
          ElevatedButton(
            child: Text('send'),
            onPressed: () async {
              if (ongoing_wifi_setup) {
                print("[BLE] Wifi setup is ongoing");
              } else {
                ongoing_wifi_setup = true;
                var result =
                    await wifi_setup(widget.deviceId, "new_ssid", "new_pass");
                print(result);
                ongoing_wifi_setup = false;
              }
            },
          ),
        ],
      ),
    );
  }
}
