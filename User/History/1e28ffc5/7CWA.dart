@override
Widget build(BuildContext context) {
  return Scaffold(
    body: Column(
      children: [
        // Custom header section
        Container(
          padding: const EdgeInsets.only(
              top: 26.0,
              right: 16.0,
              left: 16.0,
              bottom: 16.0), // Adding bottom padding
          color: Colors.transparent,
          child: Row(
            children: <Widget>[
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.only(
                      top: 20.0), // Add padding to the title specifically
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
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 30.0, vertical: 8.0),
                      title: Text(alias ?? macAddress),
                      subtitle: alias == null ? null : Text('MAC: $macAddress'),
                      onTap: () {
                        Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (context) => DummyPage()));
                      },
                      trailing: Icon(Icons.warning_amber,
                          color: Colors
                              .amber), // This line adds the icon to the right side
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
