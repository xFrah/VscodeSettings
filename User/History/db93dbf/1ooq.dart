// ignore_for_file: library_private_types_in_public_api, use_key_in_widget_constructors, prefer_const_constructors_in_immutables, constant_identifier_names, avoid_print, deprecated_member_use, non_constant_identifier_names

import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

import 'wifi_setup.dart';
import 'item_registration.dart';

// those are the values that the ESP32 sends to the app via ble

const wifi_down = 48;
const wifi_up = 49;
const wifi_connecting = 50;
const wifi_not_initialized = 51;

// color dictionary
const Color wifi_down_color = Colors.red;
const Color wifi_up_color = Colors.green;
const Color wifi_connecting_color = Colors.yellow;
const Color wifi_not_initialized_color = Colors.grey;

class ConnectedDeviceScreen extends StatefulWidget {
  final BluetoothDevice device;

  ConnectedDeviceScreen({required this.device});

  @override
  _ConnectedDeviceScreenState createState() => _ConnectedDeviceScreenState();
}

class _ConnectedDeviceScreenState extends State<ConnectedDeviceScreen> {
  /// The UUID of the wifi status characteristic
  final String wifi_status_uuid =
      "6E400003-B5A3-F393-E0A9-E50E24DCCA9E".toLowerCase();

  /// The characteristic value stream subscription
  Stream<List<int>>? characteristicValueStream;

  /// The initial color of the floating action button
  Color buttonColor = Colors.grey;

  /// This is set to true when we read the characteristic's value for the first time
  bool initialReadComplete = false;

  /// The list of services offered by the connected device
  List<BluetoothService> services = [];

  @override
  void initState() {
    super.initState();
    setupCharacteristics();
    widget.device.connectionState.then((isConnected) {
      if (!isConnected) {
        Navigator.popUntil(context, ModalRoute.withName('/'));
      }
    });

    widget.device.state.listen((state) {
      if (state == BluetoothDeviceState.disconnected) {
        Navigator.popUntil(context, ModalRoute.withName('/'));
      }
    });
  }

  /// Returns the color associated with the value read from the wifi status characteristic
  Color get_color(int read) {
    switch (read) {
      case wifi_down:
        return wifi_down_color;
      case wifi_up:
        return wifi_up_color;
      case wifi_connecting:
        return wifi_connecting_color;
      case wifi_not_initialized:
        return wifi_not_initialized_color;
      default:
        return Colors.grey;
    }
  }

  /// Enables notifications for the wifi status characteristic
  ///
  /// The characteristic value is read immediately and then the listener
  /// is set up to listen for changes to the characteristic value
  Future<void> setupCharacteristics() async {
    await widget.device.requestMtu(250);
    services = await widget.device.discoverServices();
    for (BluetoothService service in services) {
      for (BluetoothCharacteristic characteristic in service.characteristics) {
        if (characteristic.uuid.toString() == wifi_status_uuid) {
          // Read the characteristic value immediately
          List<int> initialValue;
          try {
            initialValue = await characteristic.read();
          } catch (e) {
            print(e);
            initialValue = [wifi_not_initialized];
          }
          print("[BLE] Initial value: $initialValue");

          // Indicate that the initial read is complete
          setState(() {
            // TODO it might crash
            initialReadComplete = true;
            try {
              buttonColor = get_color(initialValue[0]);
            } catch (e) {
              buttonColor = Colors.grey;
              print(e);
            }
          });

          // Then subscribe to further updates
          characteristic.setNotifyValue(true);
          characteristicValueStream = characteristic.value;
          break;
        }
      }
    }
  }

  @override
  Widget build(BuildContext context, {double circleSpacing = 45.0}) {
    final Size size = MediaQuery.of(context).size;
    final double circleRadius = size.width * 0.65; // 50% of screen width

    return Scaffold(
      appBar: AppBar(
        title: Text('Connected to ${widget.device.name}'),
      ),
      body: Padding(
        padding: EdgeInsets.symmetric(
            vertical: circleSpacing), // Add padding at the top and bottom
        child: Column(
          children: [
            Flexible(
              flex: 1,
              child: Center(
                child: Material(
                  color: Colors.transparent,
                  shape: CircleBorder(),
                  child: Ink(
                    decoration: BoxDecoration(
                      color: Colors.blue, // Replace with your color
                      shape: BoxShape.circle,
                    ),
                    child: InkWell(
                      borderRadius: BorderRadius.circular(circleRadius),
                      onTap: initialReadComplete && buttonColor != Colors.grey
                          ? () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => ReadyView(),
                                ),
                              );
                            }
                          : null, // disables the button if the condition is not met
                      splashColor: Colors.blue.withAlpha(100),
                      child: Container(
                        width: circleRadius,
                        height: circleRadius,
                        child: Icon(
                          Icons.medication_outlined,
                          color: Colors.white,
                          size: circleRadius *
                              0.5, // Modify this value to adjust the icon size
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
            Flexible(
              flex: 1,
              child: StreamBuilder<List<int>>(
                stream: characteristicValueStream,
                initialData: [],
                builder:
                    (BuildContext context, AsyncSnapshot<List<int>> snapshot) {
                  if (snapshot.hasData && snapshot.data!.isNotEmpty) {
                    int read = snapshot.data![0];
                    buttonColor = get_color(read);
                  }

                  return Center(
                    child: Material(
                      color: Colors.transparent,
                      shape: CircleBorder(),
                      child: Ink(
                        decoration: BoxDecoration(
                          color: buttonColor,
                          shape: BoxShape.circle,
                        ),
                        child: InkWell(
                          borderRadius: BorderRadius.circular(circleRadius),
                          onTap: initialReadComplete &&
                                  buttonColor != Colors.grey
                              ? () {
                                  Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (context) =>
                                          WifiCredentialsScreen(
                                        device: widget.device,
                                        services: services,
                                      ),
                                    ),
                                  );
                                }
                              : null, // disables the button if the condition is not met
                          splashColor: buttonColor.withAlpha(100),
                          child: Container(
                            width: circleRadius,
                            height: circleRadius,
                            child: Icon(
                              Icons.wifi,
                              color: Colors.white,
                              size: circleRadius *
                                  0.5, // Modify this value to adjust the icon size
                            ),
                          ),
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
