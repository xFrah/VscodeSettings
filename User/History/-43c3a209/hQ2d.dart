class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Home Screen'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            IconButton(
              icon: Icon(Icons.content_cut),
              iconSize: 50.0,
              onPressed: () {
                Navigator.pushNamed(context, '/next', arguments: 'Scissors');
              },
            ),
            IconButton(
              icon: Icon(Icons.local_hospital),
              iconSize: 50.0,
              onPressed: () {
                Navigator.pushNamed(context, '/next', arguments: 'Bandages');
              },
            ),
            RaisedButton(
              onPressed: () {
                Navigator.pushNamed(context, '/next', arguments: 'Blank 1');
              },
              child: Text(''),
              color: Colors.blue,
            ),
            RaisedButton(
              onPressed: () {
                Navigator.pushNamed(context, '/next', arguments: 'Blank 2');
              },
              child: Text(''),
              color: Colors.blue,
            ),
          ],
        ),
      ),
    );
  }
}

class NextScreen extends StatefulWidget {
  @override
  _NextScreenState createState() => _NextScreenState();
}

class _NextScreenState extends State<NextScreen> {
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
    final button = ModalRoute.of(context).settings.arguments;

    return Scaffold(
      appBar: AppBar(
        title: Text('Next Screen'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('Pressed: $button button'),
            RaisedButton(
              onPressed: _isEnabled ? disableButton : null,
              child: Text('Press me'),
              color: Colors.blue,
            ),
          ],
        ),
      ),
    );
  }
}