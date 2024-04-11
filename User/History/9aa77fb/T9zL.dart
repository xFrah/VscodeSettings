import 'package:flutter/material.dart';

class ReadyView extends StatefulWidget {
  @override
  _ReadyViewState createState() => _ReadyViewState();
}

class _ReadyViewState extends State<ReadyView> {
  bool _ready = false;

  void _readyAction() {
    setState(() {
      _ready = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _ready ? ItemRegistrationView() : Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
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
          onChanged: _buttonDisabled ? null : (String newValue) {
            setState(() {
              dropdownValue = newValue;
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
