import 'package:flutter/material.dart';
import 'registration_next.dart';

class RegistrationView extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Registration View'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            IconButton(
              icon: Icon(Icons.content_cut),
              iconSize: 50.0,
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => RegistrationNext(button: 'Scissors', key: null,)),
                );
              },
            ),
            IconButton(
              icon: Icon(Icons.local_hospital),
              iconSize: 50.0,
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => RegistrationNext(button: 'Bandages')),
                );
              },
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => RegistrationNext(button: 'Blank 1')),
                );
              },
              child: Text(''),
              // color: Colors.blue,
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => RegistrationNext(button: 'Blank 2')),
                );
              },
              child: Text(''),
              // color: Colors.blue,
            ),
          ],
        ),
      ),
    );
  }
}