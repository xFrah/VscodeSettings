import 'package:flutter/material.dart';

class ImageDisplayPage extends StatefulWidget {
  final String companyName;
  final String username;

  ImageDisplayPage({required this.companyName, required this.username});

  @override
  _ImageDisplayPageState createState() => _ImageDisplayPageState();
}

class _ImageDisplayPageState extends State<ImageDisplayPage> {
  @override
  Widget build(BuildContext context) {
    const double verticalMargin = 0.0;
    return Scaffold(
      body: Column(
        children: [
          // Custom header section
          Container(
            padding: const EdgeInsets.only(
                top: 26.0, right: 16.0, left: 16.0, bottom: 16.0),
            color: Colors.transparent,
            child: Row(
              children: <Widget>[
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.only(top: 20.0),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          widget.companyName,
                          style: const TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                        Text(
                          widget.username,
                          style: const TextStyle(
                            fontSize: 20,
                            color: Colors.white,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.account_circle,
                      size: 48, color: Colors.white),
                  onPressed: () {
                    // Handle profile icon tap here.
                  },
                ),
              ],
            ),
          ),
          // Image section
          Expanded(
            child: Center(
              child: Image.asset(
                'assets/4k.378.png', // replace with your image path
                fit: BoxFit.cover,
              ),
            ),
          ),
          // Buttons section
          Padding(
            padding: const EdgeInsets.symmetric(
                horizontal: 16.0), // Parametric horizontal padding
            child: Container(
              height: MediaQuery.of(context).size.height /
                  3, // A third of the screen height
              child: Column(
                children: [
                  Container(
                    height: buttonHeight,
                    width: double
                        .infinity, // Ensures the container takes the full width
                    margin: const EdgeInsets.symmetric(
                        vertical:
                            verticalMargin), // Vertical space between buttons
                    child: ElevatedButton(
                      onPressed: () {},
                      child: Text('Button 1'),
                      style: ElevatedButton.styleFrom(
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(
                              10), // Rounded corners for the button
                        ),
                      ),
                    ),
                  ),
                  Container(
                    height: buttonHeight,
                    width: double
                        .infinity, // Ensures the container takes the full width
                    margin: const EdgeInsets.symmetric(
                        vertical:
                            verticalMargin), // Vertical space between buttons
                    child: ElevatedButton(
                      onPressed: () {},
                      child: Text('Button 2'),
                      style: ElevatedButton.styleFrom(
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(
                              10), // Rounded corners for the button
                        ),
                      ),
                    ),
                  ),
                  Container(
                    height: buttonHeight,
                    width: double
                        .infinity, // Ensures the container takes the full width
                    margin: const EdgeInsets.symmetric(
                        vertical:
                            verticalMargin), // Vertical space between buttons
                    child: ElevatedButton(
                      onPressed: () {},
                      child: Text('Button 3'),
                      style: ElevatedButton.styleFrom(
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(
                              10), // Rounded corners for the button
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
