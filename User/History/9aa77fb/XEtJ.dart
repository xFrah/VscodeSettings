// ignore_for_file: prefer_const_literals_to_create_immutables, prefer_const_constructors

import 'package:flutter/material.dart';

class ReadyView extends StatefulWidget {
  @override
  _ReadyViewState createState() => _ReadyViewState();
}

class _ReadyViewState extends State<ReadyView> {
  bool _ready = false;
  double circleRadius =
      150.0; // You can change this to adjust the circle radius

  void _readyAction() {
    setState(() {
      _ready = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _ready
          ? ItemRegistrationView()
          : Center(
              child: Container(
                width: circleRadius * 2,
                height: circleRadius * 2,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.blue, // Circle color is now blue
                ),
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      Icon(
                        Icons
                            .sensors_outlined, // replace with your desired icon
                        size: 100.0,
                        color: Colors.white, // Icon color is now white
                      ),
                      Padding(
                        padding:
                            EdgeInsets.only(top: 20.0), // Adjust padding here
                        child: Flexible(
                          child: Text(
                            'Place item on the sensor and press Ready to continue',
                            textAlign: TextAlign.center, // Center align text
                            style: TextStyle(fontSize: 24, color: Colors.white), // Text color is now white
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
      bottomNavigationBar: !_ready
          ? BottomAppBar(
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: ElevatedButton(
                  onPressed: _readyAction,
                  child: Text('Ready'),
                ),
              ),
            )
          : null,
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

  void _saveAction() {
    setState(() {
      _buttonDisabled = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: DropdownButton<String>(
          value: dropdownValue,
          icon: Icon(Icons.arrow_downward),
          onChanged: _buttonDisabled
              ? null
              : (String? newValue) {
                  setState(() {
                    dropdownValue = newValue ?? dropdownValue;
                  });
                },
          items: <String>['One', 'Two', 'Three', 'Four']
              .map<DropdownMenuItem<String>>((String value) {
            return DropdownMenuItem<String>(
              value: value,
              child: Text(value),
            );
          }).toList(),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _buttonDisabled ? null : _saveAction,
        child: Icon(Icons.save),
      ),
    );
  }
}
