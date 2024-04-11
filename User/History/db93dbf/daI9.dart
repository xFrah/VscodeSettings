import 'package:flutter/material.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Design App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: YourScreen(),
    );
  }
}

class YourScreen extends StatelessWidget {
  final items = ['Item 1', 'Item 2', 'Item 3']; // Replace with your list

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          LargeAppBar(),
          Expanded(
            child: Column(
              children: [
                Image.network(
                    'https://via.placeholder.com/150'), // Replace with your image URL
                Expanded(
                  child: ListView.builder(
                    itemCount: items.length,
                    itemBuilder: (context, index) {
                      return ListTile(
                        title: Text(items[index]),
                        onTap: () {
                          // Handle your block click here
                          print('${items[index]} clicked!');
                        },
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class LargeAppBar extends StatefulWidget {
  @override
  _LargeAppBarState createState() => _LargeAppBarState();
}

class _LargeAppBarState extends State<LargeAppBar> {
  final battery = Battery();
  int? _batteryLevel;

  @override
  void initState() {
    super.initState();
    _getBatteryInfo();
  }

  _getBatteryInfo() async {
    final level = await battery.batteryLevel;
    setState(() {
      _batteryLevel = level;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 100,
      color: Colors.blue,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text('Device Name'), // TODO: replace with actual device name
          Text('Battery: ${_batteryLevel ?? 'Fetching'}%'),
        ],
      ),
    );
  }
}
