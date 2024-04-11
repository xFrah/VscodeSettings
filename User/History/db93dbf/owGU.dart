@override
Widget build(BuildContext context, {double circleSpacing = 45.0}) {
  final Size size = MediaQuery.of(context).size;
  final double circleRadius = size.width * 0.65; // 50% of screen width

  return Scaffold(
    appBar: AppBar(
      title: Text('Connected to ${widget.device.name}'),
    ),
    body: Padding(
      padding: EdgeInsets.symmetric(
          vertical: circleSpacing), // Add padding at the top and bottom
      child: Column(
        children: [
          Flexible(
            flex: 1,
            child: Center(
              child: Material(
                color: Colors.transparent,
                shape: CircleBorder(),
                child: Ink(
                  decoration: BoxDecoration(
                    color: Colors.blue, // Replace with your color
                    shape: BoxShape.circle,
                  ),
                  child: InkWell(
                    borderRadius: BorderRadius.circular(circleRadius),
                    onTap: initialReadComplete && buttonColor != Colors.grey
                        ? () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => ReadyView(),
                              ),
                            );
                          }
                        : null, // disables the button if the condition is not met
                    splashColor: Colors.blue.withAlpha(100),
                    child: Container(
                      width: circleRadius,
                      height: circleRadius,
                      child: Icon(
                        Icons.medication_outlined,
                        color: Colors.white,
                        size: circleRadius *
                            0.5, // Modify this value to adjust the icon size
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
          Flexible(
            flex: 1,
            child: StreamBuilder<List<int>>(
              stream: characteristicValueStream,
              initialData: [],
              builder:
                  (BuildContext context, AsyncSnapshot<List<int>> snapshot) {
                if (snapshot.hasData && snapshot.data!.isNotEmpty) {
                  int read = snapshot.data![0];
                  buttonColor = get_color(read);
                }

                return Center(
                  child: Material(
                    color: Colors.transparent,
                    shape: CircleBorder(),
                    child: Ink(
                      decoration: BoxDecoration(
                        color: buttonColor,
                        shape: BoxShape.circle,
                      ),
                      child: InkWell(
                        borderRadius: BorderRadius.circular(circleRadius),
                        onTap: initialReadComplete && buttonColor != Colors.grey
                            ? () {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) =>
                                        WifiCredentialsScreen(
                                      device: widget.device,
                                      services: services,
                                    ),
                                  ),
                                );
                              }
                            : null, // disables the button if the condition is not met
                        splashColor: buttonColor.withAlpha(100),
                        child: Container(
                          width: circleRadius,
                          height: circleRadius,
                          child: Icon(
                            Icons.wifi,
                            color: Colors.white,
                            size: circleRadius *
                                0.5, // Modify this value to adjust the icon size
                          ),
                        ),
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    ),
  );
}
