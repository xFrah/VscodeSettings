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
                        child: Container(
                          height: 50, // Adjust as needed
                          child: SingleChildScrollView(
                            child: Text(
                              'Place item on the sensor and press Ready to continue',
                              textAlign: TextAlign.center, // Center align text
                              style: TextStyle(
                                  fontSize: 24,
                                  color:
                                      Colors.white), // Text color is now white
                            ),
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
  int _selectedButtonIndex = -1;
  final int _columnsCount = 3;

  final List<Map<String, dynamic>> _items = [
    {'icon': Icons.content_cut, 'text': 'Scissors'},
    {'icon': Icons.school, 'text': 'School'},
    {'icon': Icons.work, 'text': 'Work'},
    {'icon': Icons.sports_basketball, 'text': 'Basketball'},
    {'icon': Icons.music_note, 'text': 'Music'},
    {'icon': Icons.fastfood, 'text': 'Food'},
    // Add more items as needed
  ];

  void _selectButton(int index) {
    setState(() {
      _selectedButtonIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GridView.count(
        crossAxisCount: _columnsCount,
        childAspectRatio: 1,
        children: List.generate(_items.length, (index) {
          return Padding(
            padding: EdgeInsets.all(5),
            child: ElevatedButton(
              onPressed: () => _selectButton(index),
              style: ButtonStyle(
                backgroundColor: MaterialStateProperty.resolveWith<Color>(
                  (Set<MaterialState> states) {
                    if (_selectedButtonIndex == index) return Colors.blue;
                    return Colors.grey;
                  },
                ),
                shape: MaterialStateProperty.all<RoundedRectangleBorder>(
                  RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(18.0),
                  ),
                ),
              ),
              child: FittedBox(
                fit: BoxFit.contain,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: <Widget>[
                    Icon(
                      _items[index]['icon'],
                      size: 50,
                    ),
                    Text(_items[index]['text']),
                  ],
                ),
              ),
            ),
          );
        }),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _selectedButtonIndex != -1
            ? () {
                print('Selected item: ${_items[_selectedButtonIndex]['text']}');
              }
            : null,
        child: Icon(Icons.save),
      ),
    );
  }
}
