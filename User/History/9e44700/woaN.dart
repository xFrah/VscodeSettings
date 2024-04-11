import 'package:firstaidkit_flutter/Login.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:http/http.dart' as http;
import 'package:html/parser.dart'
    as htmlParser; // Import at the top of your Dart file
import 'package:flutter_form_builder/flutter_form_builder.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart'; // if you haven't imported it yet

class ItemRegistrationPage extends StatefulWidget {
  // we need the mac address of the device to be passed to this page
  // so that we can send it to the API
  final String macAddress;

  const ItemRegistrationPage({Key? key, required this.macAddress})
      : super(key: key);

  @override
  _ItemRegistrationPageState createState() => _ItemRegistrationPageState();
}

class _ItemRegistrationPageState extends State<ItemRegistrationPage> {
  final _formKey = GlobalKey<FormBuilderState>();
  List<String> itemTypes = [
    "Guanti sterili monouso",
    "Visiera paraschizzi",
    "Soluzione cutanea iodopovidone",
    "Soluzione fisiologica",
    "Compresse garza 10x10",
    "Compresse garza 18x40",
    "Teli sterili monouso",
    "Pinzette medicazione",
    "Rete elastica",
    "Cotone idrofilo",
    "Cerotti pronti uso",
    "Rotoli cerotto",
    "Forbici",
    "Lacci emostatici",
    "Ghiaccio pronto uso",
    "Sacchetti rifiuti sanitari",
    "Termometro",
    "Misurazione pressione arteriosa",
  ];

  bool _isTypeSelected = false;
  bool _isDateSelected = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xffecf3e8),
      appBar: AppBar(
        automaticallyImplyLeading: false,
        backgroundColor: Colors.transparent,
        elevation: 0.0,
        title: const Text(
          'Beam Digital SRL',
          style: TextStyle(
            fontFamily: 'Lato',
            fontSize: 20,
            color: Color(0xff27312f),
            fontWeight: FontWeight.w700,
          ),
        ),
        actions: [
          IconButton(
            icon:
                const Icon(Icons.refresh, color: Color(0xff27312f), size: 28.0),
            onPressed: () {
              // Action for refresh button
            },
          ),
          IconButton(
            icon: const Icon(Icons.account_circle,
                color: Color(0xff27312f), size: 28.0),
            onPressed: () {
              // Action for account button
            },
          ),
          IconButton(
            icon: SvgPicture.string(_svg_yfxl45),
            onPressed: () {
              // Action for the settings SVG icon
            },
          ),
        ],
      ),
      body: Padding(
        // pad 20 everywhere but 40 on top
        padding: const EdgeInsets.fromLTRB(20, 35, 20, 20),
        child: FormBuilder(
          key: _formKey,
          child: Column(
            children: [
              FormBuilderDropdown(
                onChanged: (value) {
                  if (value != null && value.toString().isNotEmpty) {
                    setState(() {
                      _isTypeSelected = true;
                    });
                  } else {
                    setState(() {
                      _isTypeSelected = false;
                    });
                  }
                },
                name: 'type',
                decoration: InputDecoration(
                  hintText: "Select an item type",
                  hintStyle: TextStyle(
                    fontFamily: 'Lato',
                    fontSize: 18,
                    color: Colors.white, // White hint text
                    fontWeight: FontWeight.w700,
                  ),
                  filled: true,
                  fillColor: Color(0xff27312f),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10.0),
                    borderSide: BorderSide.none,
                  ),
                ),
                dropdownColor: Color(
                    0xff27312f), // Setting the dropdown items' background color
                items: itemTypes.map((type) {
                  return DropdownMenuItem(
                    value: type,
                    child: Text(
                      "$type",
                      style: TextStyle(
                        fontFamily: 'Lato',
                        fontSize: 20,
                        color: Colors.white, // White text for dropdown items
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  );
                }).toList(),
                style: const TextStyle(
                  // Style for the selected item
                  fontFamily: 'Lato',
                  fontSize: 20,
                  color: Colors.white, // White text for selected item
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 20),
              FormBuilderDateTimePicker(
                onChanged: (value) {
                  if (value != null) {
                    setState(() {
                      _isDateSelected = true;
                    });
                  } else {
                    setState(() {
                      _isDateSelected = false;
                    });
                  }
                },
                name: 'date',
                inputType: InputType.date,
                style: const TextStyle(
                  // Style for the selected date text
                  fontFamily: 'Lato',
                  fontSize: 20,
                  color: Colors.white, // White text for selected date
                  fontWeight: FontWeight.w700,
                ),
                decoration: InputDecoration(
                  hintText: "Select the expiration date",
                  hintStyle: TextStyle(
                    fontFamily: 'Lato',
                    fontSize: 18,
                    color: Color(0xffecf3e8),
                    fontWeight: FontWeight.w700,
                  ),
                  filled: true,
                  fillColor: Color(0xff27312f),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10.0),
                    borderSide: BorderSide.none,
                  ),
                ),
              ),
              const Spacer(),
              Row(
                children: [
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.all(4.0),
                      child: ElevatedButton(
                        onPressed: () {
                          Navigator.pop(context);
                        },
                        style: ButtonStyle(
                          textStyle: MaterialStateProperty.all(const TextStyle(
                            fontFamily: 'Lato',
                            fontSize: 20,
                            color: Color(0xffecf3e8),
                            fontWeight: FontWeight.w700,
                          )),
                          backgroundColor: MaterialStateProperty.all(
                              const Color(0xff27312f)),
                          shape: MaterialStateProperty.all(
                            RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10.0),
                            ),
                          ),
                          minimumSize: MaterialStateProperty.all(Size(
                              double.infinity, 70)), // Minimum height of 50
                        ),
                        child: const Text("Cancel"),
                      ),
                    ),
                  ),
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.all(4.0),
                      child: ElevatedButton(
                        onPressed: (_isTypeSelected && _isDateSelected)
                            ? () {
                                if (_formKey.currentState?.saveAndValidate() ?? false) {
                                  var values = _formKey.currentState?.value;
                                  print(values);
                                }
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                      builder: (context) => SensorPage(
                                          macAddress: widget.macAddress,
                                          itemType: _formKey.currentState?.value['type'] ?? '')),
                                );
                              }
                            : null,
                        With these changes, the "Save" button will only be enabled when both the dropdown has a selected item and the date has been selected.





,
                        style: ButtonStyle(
                          textStyle: MaterialStateProperty.all(const TextStyle(
                            fontFamily: 'Lato',
                            fontSize: 20,
                            color: Color(0xffecf3e8),
                            fontWeight: FontWeight.w700,
                          )),
                          backgroundColor: MaterialStateProperty.all(
                              const Color(0xff27312f)),
                          shape: MaterialStateProperty.all(
                            RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10.0),
                            ),
                          ),
                          minimumSize: MaterialStateProperty.all(Size(
                              double.infinity, 70)), // Minimum height of 50
                        ),
                        child: const Text("Save"),
                      ),
                    ),
                  ),
                ],
              )
            ],
          ),
        ),
      ),
    );
  }
}

class SensorPage extends StatefulWidget {
  final String macAddress;
  final String itemType;

  const SensorPage({Key? key, required this.macAddress, required this.itemType})
      : super(key: key);

  @override
  _SensorPageState createState() => _SensorPageState();
}

class _SensorPageState extends State<SensorPage> {
  int? _lastStatusCode;
  String _lastServerMessage = '';
  final storage = new FlutterSecureStorage();
  int _countdown = 5;
  bool _isButtonEnabled = false;
  bool _isRequestInProgress = false; // Add this

  _startCountdown() {
    Future.delayed(Duration(seconds: 1), () {
      if (_countdown > 0) {
        setState(() {
          _countdown--;
        });
        _startCountdown();
      } else {
        setState(() {
          _isButtonEnabled = true;
        });
      }
    });
  }

  @override
  void initState() {
    super.initState();
    _startCountdown();
  }

  Future<void> _registerItem() async {
    setState(() {
      _isRequestInProgress = true;
    });

    String? token = await storage.read(key: 'jwt');
    var url = Uri.parse(
        'http://5.196.23.212:8080/register_item'); // replace with your server address
    var formattedType = widget.itemType.replaceAll(' ', '_').toLowerCase();

    var response = await http.post(url, headers: {
      'Authorization': 'Bearer $token',
    }, body: {
      'kit_mac_address':
          widget.macAddress, // replace with actual value or variable
      'item_type': formattedType, // replace with actual value or variable
    }).timeout(const Duration(seconds: 30));

    var document = htmlParser.parse(response.body);
    var pContent =
        document.querySelector('p')?.text ?? 'Error: Content not found';

    setState(() {
      _isRequestInProgress = false;
      _lastStatusCode = response.statusCode;
      _lastServerMessage = pContent;
    });

    if (response.statusCode == 200 || response.statusCode == 401) {
      await Future.delayed(Duration(seconds: 5)); // Wait for 5 seconds
    }

    if (response.statusCode == 200) {
      Navigator.pop(context);
    } else if (response.statusCode == 401) {
      Navigator.pushReplacement(
          context, MaterialPageRoute(builder: (context) => Login()));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xffecf3e8),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Expanded(
            child: Center(
              child:
                  Text("Place the item on the sensor, then press the button"),
            ),
          ),
          if (_lastStatusCode != null && _isRequestInProgress == false)
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16.0, vertical: 10.0),
              child: _ServerResponse(
                statusCode: _lastStatusCode!,
                message: _lastServerMessage,
              ),
            ),
          Padding(
            padding: const EdgeInsets.only(bottom: 20.0), // added this padding
            child: Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16.0, vertical: 10.0),
              child: ElevatedButton(
                onPressed: _isButtonEnabled && !_isRequestInProgress
                    ? _registerItem
                    : null, // Adjusted condition
                style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 25.0),
                    backgroundColor:
                        const Color(0xff27312f), // Background color
                    minimumSize: const Size(0, 88.0), // Minimum height of 88.0
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10.0))),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Expanded(
                        child: Text(
                      _isButtonEnabled ? "Register" : "$_countdown s",
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontFamily: 'Lato',
                        fontSize: 20,
                        color: Color(0xffecf3e8),
                        fontWeight: FontWeight.w700,
                      ),
                    )),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ServerResponse extends StatelessWidget {
  final int statusCode;
  final String message;

  const _ServerResponse({required this.statusCode, required this.message});

  @override
  Widget build(BuildContext context) {
    Color borderColor;
    Color backgroundColor;

    if (statusCode == 200) {
      borderColor = Colors.green;
      backgroundColor = Colors.green[100]!;
    } else {
      borderColor = Colors.red;
      backgroundColor = Colors.red[100]!;
    }

    return Container(
      padding: const EdgeInsets.all(20.0),
      decoration: BoxDecoration(
        color: backgroundColor,
        border: Border.all(color: borderColor, width: 2),
        borderRadius: BorderRadius.circular(10.0),
      ),
      child: Text(
        'Error $statusCode: $message',
        style: const TextStyle(fontWeight: FontWeight.bold),
      ),
    );
  }
}

const String _svg_yfxl45 =
    '<svg viewBox="3.0 2.4 21.4 22.0" ><path  d="M 23.9393253326416 15.52492904663086 L 23.9168758392334 15.5067777633667 L 22.40943336486816 14.32461071014404 C 22.21578407287598 14.17141723632812 22.1074104309082 13.93459892272949 22.11807060241699 13.68791198730469 L 22.11807060241699 13.13575744628906 C 22.10836601257324 12.89063358306885 22.21692276000977 12.65575408935547 22.40991020202637 12.50431251525879 L 23.9168758392334 11.3216667175293 L 23.9393253326416 11.30351543426514 C 24.41935348510742 10.90363121032715 24.5386905670166 10.2147102355957 24.22113418579102 9.676664352416992 L 22.18111991882324 6.146877765655518 C 22.17877006530762 6.143547058105469 22.17669486999512 6.140033721923828 22.17491149902344 6.136370182037354 C 21.85595893859863 5.606298923492432 21.20416450500488 5.379325866699219 20.62495994567871 5.596632957458496 L 20.60823822021484 5.602842330932617 L 18.836181640625 6.315963268280029 C 18.60928535461426 6.40770435333252 18.35211944580078 6.384777545928955 18.14503288269043 6.254347801208496 C 17.98836517333984 6.155634880065918 17.82915115356445 6.062335014343262 17.66738891601562 5.974448680877686 C 17.45468902587891 5.859107971191406 17.30947685241699 5.649534702301025 17.27619743347168 5.409874439239502 L 17.00919532775879 3.518884181976318 L 17.00346565246582 3.484493732452393 C 16.88365745544434 2.880797624588013 16.35696792602539 2.443815469741821 15.741530418396 2.4375 L 11.65672492980957 2.4375 C 11.0316333770752 2.439500331878662 10.49854469299316 2.890762567520142 10.39335918426514 3.506943225860596 L 10.38906002044678 3.533691644668579 L 10.12301254272461 5.428502082824707 C 10.08992099761963 5.667430877685547 9.94597339630127 5.876713275909424 9.734688758850098 5.993076324462891 C 9.572457313537598 6.080453395843506 9.413156509399414 6.173168182373047 9.257045745849609 6.271062850952148 C 9.050335884094238 6.400691032409668 8.793978691101074 6.423254013061523 8.567804336547852 6.331725597381592 L 6.794315338134766 5.615260601043701 L 6.777597904205322 5.608573913574219 C 6.197518825531006 5.391016006469727 5.544827461242676 5.618895530700684 5.226212024688721 6.150221347808838 L 5.2200026512146 6.16072940826416 L 3.177122354507446 9.692901611328125 C 2.859086513519287 10.23149967193604 2.978425264358521 10.92123508453369 3.458932399749756 11.3216667175293 L 3.481381416320801 11.33981704711914 L 4.988823413848877 12.52198505401611 C 5.182473659515381 12.67517852783203 5.290844917297363 12.91199684143066 5.280186176300049 13.15868282318115 L 5.280186176300049 13.71084022521973 C 5.289889335632324 13.95596218109131 5.181332588195801 14.19084167480469 4.988345623016357 14.34228515625 L 3.481380939483643 15.52492904663086 L 3.458931684494019 15.54307842254639 C 2.978901624679565 15.94296264648438 2.859563827514648 16.63188552856445 3.177121639251709 17.1699333190918 L 5.217136859893799 20.6997184753418 C 5.219484806060791 20.70304870605469 5.221561431884766 20.70656204223633 5.22334623336792 20.71022415161133 C 5.542296409606934 21.24029541015625 6.194091320037842 21.46726989746094 6.773300170898438 21.24996376037598 L 6.790016651153564 21.24375152587891 L 8.560639381408691 20.53063011169434 C 8.787535667419434 20.43889045715332 9.044700622558594 20.46181678771973 9.251791000366211 20.59224700927734 C 9.40845775604248 20.6912784576416 9.567670822143555 20.78457641601562 9.729433059692383 20.87214660644531 C 9.942132949829102 20.98748588562012 10.08734512329102 21.19705772399902 10.1206226348877 21.43671798706055 L 10.38619232177734 23.32771110534668 L 10.39192390441895 23.36210060119629 C 10.51195049285889 23.96684265136719 11.04020690917969 24.40413284301758 11.65672492980957 24.40909767150879 L 15.741530418396 24.40909576416016 C 16.36662101745605 24.40709495544434 16.89971160888672 23.95583152770996 17.00489616394043 23.33964920043945 L 17.00919723510742 23.31290245056152 L 17.27524566650391 21.41809272766113 C 17.30885696411133 21.17869758605957 17.45375633239746 20.96932029724121 17.66595649719238 20.85351753234863 C 17.82931137084961 20.76563262939453 17.98884391784668 20.6724910736084 18.14359855651855 20.57552909851074 C 18.35031127929688 20.44590187072754 18.60666847229004 20.42334175109863 18.83284187316895 20.51487159729004 L 20.6063289642334 21.22894668579102 L 20.62304878234863 21.23563194274902 C 21.20312309265137 21.4536018371582 21.8560791015625 21.22562980651855 22.17443466186523 20.6939868927002 C 22.17632484436035 20.69038200378418 22.17839622497559 20.68687438964844 22.18063926696777 20.68347930908203 L 24.22065734863281 17.15417289733887 C 24.5392894744873 16.6156177520752 24.42011070251465 15.92544746398926 23.9393253326416 15.52493190765381 Z M 17.51597785949707 13.60289096832275 C 17.41903495788574 15.66443824768066 15.70261383056641 17.2764835357666 13.63904571533203 17.24407768249512 C 11.57547855377197 17.211669921875 9.910527229309082 15.54651927947998 9.878366470336914 13.48294639587402 C 9.846205711364746 11.41937351226807 11.45845794677734 9.70314884185791 13.5200138092041 9.606450080871582 C 14.59368801116943 9.559181213378906 15.63784885406494 9.965047836303711 16.39774322509766 10.72503185272217 C 17.15763473510742 11.48501396179199 17.56337738037109 12.52922248840332 17.5159797668457 13.60289096832275 Z" fill="#27312f" stroke="none" stroke-width="0.09375" stroke-miterlimit="4" stroke-linecap="butt" /></svg>';
