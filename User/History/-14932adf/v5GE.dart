import 'dart:convert';
import 'package:firstaidkit_flutter/Home.dart';
import 'package:firstaidkit_flutter/Login.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'kit_panel.dart';

const storage = FlutterSecureStorage();

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Login App',
      // theme: ThemeData.dark(),
      home: Login(),
    );
  }
}
