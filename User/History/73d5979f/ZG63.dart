import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:logging/logging.dart';
import 'package:quick_blue/quick_blue.dart';

import 'PeripheralDetailPage.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  StreamSubscription<BlueScanResult>? _subscription;

  @override
  void initState() {
    super.initState();
    if (kDebugMode) {
      QuickBlue.setLogger(Logger('quick_blue_example'));
    }
    _subscription = QuickBlue.scanResultStream.listen((result) {
      // print in console the advertisement data
      print("cock");
      print(result.manufacturerData);
      print("cock2");
      print(result.name);
      if (!_scanResults.any((r) => r.deviceId == result.deviceId)) {
        if (result.name == "First-Aid-Kit") {
          setState(() => _scanResults.add(result));
        }
      }
    });
  }

  @override
  void dispose() {
    super.dispose();
    _subscription?.cancel();
  }

  void wifi_setup(String deviceId, String new_ssid, String new_pass) {
    final String cmd_service = "";
    final String rx_characteristic = "";
    List<int> list = "command:wifi_setup".codeUnits;
    Uint8List bytes = Uint8List.fromList("command:wifi_setup".codeUnits);
    final Uint8List wifi_setup_cmd = "command:wifi_setup";
    
    // write command:wifi_setup to gatt rx charateristic
    QuickBlue.writeValue(deviceId, cmd_service, rx_characteristic, wifi_setup_cmd, bleOutputProperty)
    // wait for confirmation
    // write ssid and password to the respective charateristics
    // wait for confirmation
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Plugin example app'),
        ),
        body: Column(
          children: [
            FutureBuilder(
              future: QuickBlue.isBluetoothAvailable(),
              builder: (context, snapshot) {
                var available = snapshot.data?.toString() ?? '...';
                return Text('Bluetooth init: $available');
              },
            ),
            _buildButtons(),
            Divider(
              color: Colors.blue,
            ),
            _buildListView(),
            _buildPermissionWarning(),
          ],
        ),
      ),
    );
  }

  Widget _buildButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: <Widget>[
        ElevatedButton(
          child: Text('startScan'),
          onPressed: () {
            QuickBlue.startScan();
          },
        ),
        ElevatedButton(
          child: Text('stopScan'),
          onPressed: () {
            QuickBlue.stopScan();
          },
        ),
      ],
    );
  }

  var _scanResults = <BlueScanResult>[];

  Widget _buildListView() {
    return Expanded(
      child: ListView.separated(
        itemBuilder: (context, index) => ListTile(
          title:
              Text('${_scanResults[index].name}(${_scanResults[index].rssi})'),
          subtitle: Text(_scanResults[index].deviceId),
          onTap: () {
            Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) =>
                      PeripheralDetailPage(_scanResults[index].deviceId),
                ));
          },
        ),
        separatorBuilder: (context, index) => Divider(),
        itemCount: _scanResults.length,
      ),
    );
  }

  Widget _buildPermissionWarning() {
    if (Platform.isAndroid) {
      return Container(
        margin: EdgeInsets.symmetric(horizontal: 10),
        child: Text('BLUETOOTH_SCAN/ACCESS_FINE_LOCATION needed'),
      );
    }
    return Container();
  }
}
