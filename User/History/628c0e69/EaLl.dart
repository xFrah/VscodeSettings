import 'package:flutter/material.dart';

class RegistrationNext extends StatefulWidget {
  final String button;

  RegistrationNext({required Key key, required this.button}) : super(key: key);

  @override
  _RegistrationNextState createState() => _RegistrationNextState();
}

class _RegistrationNextState extends State<RegistrationNext> {
  bool _isEnabled = true;

  void disableButton() {
    setState(() {
      _isEnabled = false;
    });
    Future.delayed(Duration(seconds: 5), () {
      setState(() {
        _isEnabled = true;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Registration Next'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('Pressed: ${widget.button} button'),
            ElevatedButton(
              onPressed: _isEnabled ? disableButton : null,
              child: Text('Press me'),
              // color: Colors.blue,
            ),
          ],
        ),
      ),
    );
  }
}