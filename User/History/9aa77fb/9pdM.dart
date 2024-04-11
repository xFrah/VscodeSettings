import 'package:flutter/material.dart';

class ReadyView extends StatefulWidget {
  @override
  _ReadyViewState createState() => _ReadyViewState();
}

class _ReadyViewState extends State<ReadyView> {
  bool _ready = false;
  double circleRadius = 150.0; // You can change this to adjust the circle radius

  void _readyAction() {
    setState(() {
      _ready = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _ready ? ItemRegistrationView() : Center(
        child: Container(
          width: circleRadius * 2,
          height: circleRadius * 2,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: Colors.lightBlue[50],  // Replace with your desired color
          ),
          child: Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: <Widget>[
                Icon(
                  Icons.ac_unit, // replace with your desired icon
                  size: 100.0,
                ),
                Text(
                  'Your Text Here',
                  style: TextStyle(fontSize: 24),
                ),
              ],
            ),
          ),
        ),
      ),
      bottomNavigationBar: !_ready ? BottomAppBar(
        child: Padding(
          padding: const EdgeInsets.all(8.0),
          child: ElevatedButton(
            onPressed: _readyAction,
            child: Text('Ready'),
          ),
        ),
      ) : null,
    );
  }
}

class ItemRegistrationView extends StatefulWidget {
  @override
  _ItemRegistrationViewState createState() => _ItemRegistrationViewState();
}

class _ItemRegistrationViewState extends State<ItemRegistrationView> {
  bool _buttonDisabled = false;
  String dropdownValue = 'One';

  void _saveAction()
