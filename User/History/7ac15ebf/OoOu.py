from datetime import datetime


def parse_I(packet_bytes):
    packet_structure = [
        ("STX", 1),
        ("Terminal Identifier", 8),
        ("Fixed Value", 1),
        ("Command Code", 1),
        ("ETX", 1),
        ("LRC", 1),
    ]

    result = {}
    pointer = 0

    for field_name, field_length in packet_structure:
        field_data = packet_bytes[pointer : pointer + field_length]
        pointer += field_length

        if field_name in ["STX", "ETX", "LRC"]:
            result[field_name] = field_data.hex().upper()
        else:
            result[field_name] = field_data.decode("ascii", errors="ignore")

    return result


def parse_F(packet_bytes):
    # Define the packet structure
    packet_structure = [
        ("STX", 1),
        ("Terminal Identifier", 8),
        ("Fixed Value", 1),
        ("Command Code", 1),
        ("Error Code", 2),
        ("Error Message - 1st line", 16),
        ("Error Message - 2nd line", 16),
        ("ETX", 1),
        ("LRC", 1),
    ]

    # Error code interpretation (example, needs to be filled with actual error codes)
    error_code_interpretation = {
        "00": "No Error",
        "01": "General Error",
        # Add other error codes based on Table 2 mentioned in the documentation
    }

    # Initialize the pointer and the result dictionary
    pointer = 0
    result = {}

    # Iterate over the packet structure and parse fields
    for field_name, field_length in packet_structure:
        # Extract the field data
        field_data = packet_bytes[pointer : pointer + field_length]
        # Increment the pointer
        pointer += field_length

        # Handle hexadecimal fields
        if field_name in ["STX", "ETX", "LRC"]:
            result[field_name] = field_data.hex().upper()
        else:
            # Decode ASCII fields
            result[field_name] = field_data.decode("ascii", errors="ignore").strip()

        # Interpret fields that have a dictionary of interpretations
        if field_name == "Error Code":
            result[field_name] = error_code_interpretation.get(
                result[field_name], "Unknown Error Code"
            )

    # Return the parsed and interpreted packet
    return result


def parse_ads(packet_bytes):
    # Define the expected packet fields and their lengths
    packet_structure = [
        ("STX", 1),  # Start of Text
        ("Terminal Identifier", 8),  # ASCII representation expected
        ("Fixed Value", 1),  # ASCII representation expected
        ("Operation Code", 1),  # ASCII representation expected
        ("Terminal Status", 1),  # ASCII representation expected
        ("Card Status", 1),  # ASCII representation expected
        ("Command Result", 1),  # ASCII representation expected
        ("ETX", 1),  # End of Text
        ("LRC", 1),  # Longitudinal Redundancy Check
    ]

    parsed_data = {}
    pointer = 0

    for field_name, field_length in packet_structure:
        field_data = packet_bytes[pointer : pointer + field_length]
        pointer += field_length

        # For human readability, convert bytes to their appropriate representation
        if field_name in ["STX", "ETX", "LRC"]:
            # Convert control characters to their hexadecimal representation
            parsed_data[field_name] = field_data.hex().upper()
        else:
            # Convert other fields to ASCII
            parsed_data[field_name] = field_data.decode("ascii", errors="ignore")

    # Provide human-readable interpretations for specific fields
    # Note: These interpretations should be expanded based on full specification
    terminal_status_interpretations = {
        "0": "CB2 keys not present",
        "1": "no banking parameters present; FIRST DLL needed",
        "2": "terminal is blocked; call maintenance",
        "3": "terminal not operative; acquirer parameters missing",
        "4": "terminal is ready and active",
        "5": "terminal is ready and NOT active",
        "6": "log full",
    }

    card_status_interpretations = {
        "0": "the card is in the terminal",
        "1": "no card present in the terminal",
        "2": "RFU",
    }

    command_result_interpretations = {
        "1": "OK",
        "2": "KO",
    }

    # Apply interpretations if available
    parsed_data["Terminal Status"] = terminal_status_interpretations.get(
        parsed_data["Terminal Status"], "Unknown Status"
    )
    parsed_data["Card Status"] = card_status_interpretations.get(
        parsed_data["Card Status"], "Unknown Status"
    )
    parsed_data["Command Result"] = command_result_interpretations.get(
        parsed_data["Command Result"], "Unknown Result"
    )

    return parsed_data


def parse_E(packet_bytes):
    # Define the packet structure
    packet_structure = [
        ("STX", 1),
        ("Terminal Identifier", 8),
        ("Fixed Value", 1),
        ("Command Code", 1),
        ("Transaction Result", 2),
        ("Acquirer ID", 11),
        ("Transaction Type", 3),
        ("Ticket Number Echo", 6),
        ("Card Type", 1),
        ("STAN", 6),
        ("Approved or Authorized Amount", 8),
        ("Transaction Date and Time", 12),
        ("Approval Type", 1),
        ("Acquirer Name", 16),
        ("PAN", 19),
        ("RFU", 5),
        ("Receipt rows", 120),
        ("Message for POS", 16),
        ("Approval Code", 6),
        ("Merchant Identifier", 15),
        ("Issuer Code", 5),
        ("Action Code", 3),
        ("Authorization Response Code", 2),
        ("Operation Type", 1),
        ("Transaction Identifier", 2),
        ("ETX", 1),
        ("LRC", 1),
    ]

    # Dictionary to interpret certain fields
    transaction_result_interpretation = {
        "00": "Approved"
        # Add other interpretations as necessary
    }
    card_type_interpretation = {
        "0": "National Debit Card",
        "1": "International Card"
        # Add other interpretations as necessary
    }
    approval_type_interpretation = {
        "0": "Offline",
        "1": "Online"
        # Add other interpretations as necessary
    }

    # Initialize the pointer and the result dictionary
    pointer = 0
    result = {}

    # Iterate over the packet structure and parse fields
    for field_name, field_length in packet_structure:
        # Extract the field data
        field_data = packet_bytes[pointer : pointer + field_length]
        # Increment the pointer
        pointer += field_length

        # Handle hexadecimal fields
        if field_name in ["STX", "ETX", "LRC"]:
            result[field_name] = field_data.hex().upper()
        else:
            # Decode ASCII fields
            result[field_name] = field_data.decode("ascii", errors="ignore").strip()

        # Interpret fields that have a dictionary of interpretations
        if field_name == "Transaction Result":
            result[field_name] = transaction_result_interpretation.get(
                result[field_name], "Unknown"
            )
        elif field_name == "Card Type":
            result[field_name] = card_type_interpretation.get(
                result[field_name], "Unknown"
            )
        elif field_name == "Approval Type":
            result[field_name] = approval_type_interpretation.get(
                result[field_name], "Unknown"
            )

    # Return the parsed and interpreted packet
    return result


def pos_server1(transaction_id, success, cart_id):
    return {
        "packet_type": "pos-server1",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": {"transaction_id": transaction_id, "state": "TRANSAZIONE"},
        "success": success,
        "data": {"cart_id": cart_id},
    }


def pos_server2(transaction_id, success):
    return {
        "packet_type": "pos-server2",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": {"transaction_id": transaction_id, "state": "IDLE"},
        "success": success,
    }


def ack(msg, status):
    return {
        "packet_type": "ack",
        "seq_num": msg["seq_num"],
        "request_type": msg["packet_type"],
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": {"transaction_id": msg["transaction_id"], "state": status},
    }


def stampante_server(transaction_id, success, printer_status):
    return {
        "packet_type": "stampante-server",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": {"transaction_id": transaction_id, "state": "IDLE"},
        "success": success,
        "data": {"printedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        "printer_status": printer_status,
    }


# Note: To use this function, you need to provide the actual packet byte string.
# The example_packet variable must be replaced with the actual data to parse.
# Here's how you would call the function:
# actual_packet_data = b'...'  # Your byte string here
# parsed_packet = parse_extended_packet(actual_packet_data)
# for key, value in parsed_packet.items():
#     print(f"{key}: {value}")

if __name__ == "__main__":
    example_packet = b"\x02925411880E0000000000040CLI0000001000004000000101011232232361Mastercard      ************1661   00000*                        ANTIQUARIUM CANNE BATTAGSP 142 CANNE DELLA BATTAARRIVEDERCI E GRAZIE    Cod. Commerc: 22537427                  N5A9KP493461400040002     00000000\x03\x01"
    msg = parse_E(example_packet)
    # pretty print the parsed packet
    for key, value in msg.items():
        print(f"{key}: {value}")
