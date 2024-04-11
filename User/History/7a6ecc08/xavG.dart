import 'package:firstaidkit_flutter/Calilbration.dart';
import 'package:firstaidkit_flutter/wifi_setup.dart';
import 'package:flutter/material.dart';

class SettingsPage extends StatelessWidget {
  final String macAddress;

  SettingsPage({required this.macAddress});

  late Map<String, Widget> pages = {
    'Wifi Setup': WifiCredentialsScreen(device: null, services: [],),
    'Calibration': BLECalibration(macAddress: macAddress),
  };


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Settings'),
      ),
      body: ListView.builder(
        itemCount: pages.length,
        itemBuilder: (context, index) {
          final title = pages.keys.elementAt(index);
          return ListTile(
            title: Text(title),
            trailing: Icon(Icons.arrow_forward_ios),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => pages[title]!,
                ),
              );
            },
          );
        },
      ),
    );
  }
}