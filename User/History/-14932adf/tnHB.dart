import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

final storage = FlutterSecureStorage();

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Login App',
      theme: ThemeData.dark(),
      home: LoginScreen(),
    );
  }
}

class LoginScreen extends StatefulWidget {
  @override
  _LoginScreenState createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  bool _isLoading = false;

  Future<void> _login() async {
    setState(() {
      _isLoading = true; // Set loading to true when login starts
    });

    final response = await http.post(
      Uri.parse('http://5.196.23.212:8080/access'),
      body: {
        'username': _usernameController.text,
        'password': _passwordController.text,
      },
    );

    if (response.statusCode == 200) {
      Map<String, dynamic> data = json.decode(response.body);
      await storage.write(key: 'jwt', value: data['access_token']);
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => DeviceListScreen()),
      );
    } else {
      // Bounce back timer (for example, 2 seconds)
      await Future.delayed(const Duration(seconds: 2));
      if (mounted) {
        // Ensure the widget is still in the tree
        setState(() {
          _isLoading = false; // Reset loading after delay
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
                controller: _usernameController,
                decoration: const InputDecoration(labelText: 'Username')),
            TextField(
                controller: _passwordController,
                decoration: const InputDecoration(labelText: 'Password'),
                obscureText: true),
            ElevatedButton(
              onPressed: _isLoading
                  ? null
                  : _login, // Disable button if _isLoading is true
              child: _isLoading
                  ? const Text('Logging in...')
                  : const Text('Login'),
            )
          ],
        ),
      ),
    );
  }
}

class DeviceListScreen extends StatefulWidget {
  final String companyName = "TechCorp";
  final String username = "JohnDoe";
  @override
  _DeviceListScreenState createState() => _DeviceListScreenState();
}

class _DeviceListScreenState extends State<DeviceListScreen> {
  //Map<String, dynamic> devices = {};
  List<MapEntry<String, dynamic>> devicesList = [];
  double appBarHeight = 100.0; // or another height you prefer

  @override
  void initState() {
    super.initState();
    _fetchDevices();
  }

  Future<void> _fetchDevices() async {
    String? token = await storage.read(key: 'jwt');
    final response = await http.get(
      Uri.parse('http://5.196.23.212:8080/get_devices'),
      headers: {
        'Authorization': 'Bearer $token',
      },
    );

    if (response.statusCode == 200) {
      setState(() {
        devicesList = jsonDecode(response.body).entries.toList();
        // turn map into list of key-value pairs
        // devicesList = devices.entries.toList();
        // print in console
        print("cock");
        // print(devices);
      });
    } else {
      // TODO: Handle the error appropriately
    }
  }

  Future<void> _refreshList() async {
    await _fetchDevices();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // Custom header section
          Container(
            padding: const EdgeInsets.only(
                top: 25.0,
                right: 16.0,
                left: 16.0,
                bottom: 16.0), // Adding bottom padding
            color: Colors.transparent,
            child: Row(
              children: <Widget>[
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.only(
                        top: 14.0), // Add padding to the title specifically
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
                      size: 48, color: Colors.white), // Adjust size accordingly
                  onPressed: () {
                    Navigator.push(context,
                        MaterialPageRoute(builder: (context) => DummyPage()));
                  },
                ),
              ],
            ),
          ),
          // Device list section
          Expanded(
            child: RefreshIndicator(
              onRefresh: _refreshList,
              child: ListView.builder(
                itemCount: devicesList.length,
                itemBuilder: (context, index) {
                  String macAddress = devicesList[index].key;
                  String? alias = devicesList[index].value;

                  return Padding(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 8.0,
                        vertical: 4.0), // To add space between the cards
                    child: Card(
                      shape: RoundedRectangleBorder(
                        borderRadius:
                            BorderRadius.circular(10.0), // Rounded corners
                      ),
                      color: Color.fromRGBO(
                          255, 255, 255, 0.08), // Semi-transparent white
                      child: ListTile(
                        contentPadding: EdgeInsets.symmetric(
                            horizontal: 16.0,
                            vertical: 8.0), // Add padding inside ListTile
                        title: Text(alias ?? macAddress),
                        subtitle:
                            alias == null ? null : Text('MAC: $macAddress'),
                        onTap: () {
                          Navigator.push(
                              context,
                              MaterialPageRoute(
                                  builder: (context) => DummyPage()));
                        },
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class DummyPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Dummy Page')),
      body: const Center(child: Text('Dummy content')),
    );
  }
}
