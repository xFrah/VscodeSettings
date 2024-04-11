import 'package:firstaidkit_flutter/Calilbration.dart';
import 'package:flutter/material.dart';

class SettingsPage extends StatelessWidget {
  final String macAddress;

  SettingsPage({required this.macAddress});

  final Map<String, Widget> pages = {
    'Wifi Setup': WifiSetupPage(),
    'Calibration': BLECalibration(macAddress: ,),
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