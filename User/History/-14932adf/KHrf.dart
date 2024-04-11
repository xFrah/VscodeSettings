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

  Future<void> _login() async {
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
      // TODO: Show an error message or handle the error appropriately
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
            ElevatedButton(onPressed: _login, child: const Text('Login'))
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
  Map<String, dynamic> devices = {};
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
        devices = jsonDecode(response.body);
        // turn map into list of key-value pairs
        devicesList = devices.entries.toList();
        // print in console
        print("cock");
        print(devices);
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
              child: ListView.builder(
            itemCount: devicesList.length,
            itemBuilder: (context, index) {
              String macAddress = devicesList[index].key;
              String alias = devicesList[index].value;

              return ListTile(
                title: Text(alias ?? macAddress),
                subtitle: alias == null ? null : Text('MAC: $macAddress'),
                onTap: () {
                  Navigator.push(context,
                      MaterialPageRoute(builder: (context) => DummyPage()));
                },
              );
            },
          )),
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
