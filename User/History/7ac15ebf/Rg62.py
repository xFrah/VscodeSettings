def parse_F(packet_bytes):
    # Define the expected packet fields and their lengths from the new structure
    packet_structure = [
        ("STX", 1),  # Start of Text
        ("Terminal Identifier", 8),  # ASCII representation expected
        ("Fixed Value", 1),  # ASCII representation expected
        ("Operation Code", 1),  # ASCII representation expected
        ("Error Code", 2),  # ASCII representation expected
        ("Error Message - 1st line", 16),  # ASCII representation expected
        ("Error Message - 2nd line", 16),  # ASCII representation expected
        ("Warning: card in", 1),  # ASCII representation expected
        ("Fixed Value 2", 4),  # ASCII representation expected
        ("ETX", 1),  # End of Text
        ("LRC", 1),  # Longitudinal Redundancy Check
    ]

    parsed_data = {}
    pointer = 0

    for field_name, field_length in packet_structure:
        field_data = packet_bytes[pointer : pointer + field_length]
        pointer += field_length

        # Convert bytes to their appropriate representation
        if field_name in ["STX", "ETX", "LRC"]:
            # Convert control characters to their hexadecimal representation
            parsed_data[field_name] = field_data.hex().upper()
        else:
            # Convert other fields to ASCII, stripping any null space padding
            parsed_data[field_name] = field_data.decode(
                "ascii", errors="ignore"
            ).rstrip()

    # Return the parsed packet data
    return parsed_data


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
    # Define the expected packet fields and their lengths from the structure
    packet_structure = [
        ("STX", 1),
        ("Terminal Identifier", 8),
        ("Fixed Value", 1),
        ("Operation Code", 1),
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
        ("Warning: card in", 1),
        ("Service Id", 4),
        ("Pre-authorization code", 12),
        ("RFU", 1),
        ("Receipt rows", 120),
        ("Message for POS", 16),
        ("Approval Code", 6),
        ("Merchant Identifier", 15),
        ("Issuer Code", 5),
        ("Action Code", 3),
        ("Authorization Response Code", 2),
        ("Operation Type", 1),
        ("Transaction Identifier", 2),
        ("AID_L14", 14),
        ("TVR", 10),
        ("AC", 16),
        ("IAD", 64),
        ("ARC", 2),
        ("APPL LABEL", 16),
        ("ATC", 4),
        ("TCC", 3),
        ("TT", 2),
        ("TrCC", 3),
        ("UN", 8),
        ("TSI", 4),
        ("TAC", 30),
        ("CVMR", 6),
        ("AUC", 4),
        ("AIP", 4),
        ("IAC", 30),
        ("CID", 2),
        ("OPS", 1),
        ("ApplPN", 16),
        ("CTQ", 4),
        ("AID", 16),
        ("ETX", 1),
        ("LRC", 1),
    ]

    # Interpretation dictionaries for fields with specific meanings
    transaction_result_codes = {
        "00": "Approved",
        # Add more codes as necessary
    }
    card_type_codes = {
        "0": "National Debit Card",
        "1": "International Credit Card",
        # Add more codes as necessary
    }
    approval_type_codes = {
        "0": "Offline",
        "1": "Online",
        # Add more codes as necessary
    }

    # Interpretation dictionaries for fields with specific meanings, such as OPS
    ops_codes = {
        "0": "Transaction completed OFFLINE",
        "1": "Transaction completed ONLINE",
        "2": "Transaction completed as UnableToGoOnline",
    }

    parsed_data = {}
    pointer = 0

    for field_name, field_length in packet_structure:
        if pointer >= len(packet_bytes):
            break  # Avoid reading beyond the end of the packet

        field_data = packet_bytes[pointer : pointer + field_length]
        pointer += field_length

        # Convert bytes to their appropriate representation
        if field_name in ["STX", "ETX", "LRC"]:
            parsed_data[field_name] = field_data.hex().upper()
        else:
            field_value = field_data.decode("ascii", errors="ignore").rstrip()
            if field_name == "Transaction Result":
                parsed_data[field_name] = transaction_result_codes.get(
                    field_value, "Unknown Result"
                )
            elif field_name == "Card Type":
                parsed_data[field_name] = card_type_codes.get(
                    field_value, "Unknown Card Type"
                )
            elif field_name == "Approval Type":
                parsed_data[field_name] = approval_type_codes.get(
                    field_value, "Unknown Approval Type"
                )
            elif field_name == "OPS":
                parsed_data[field_name] = ops_codes.get(field_value, "Unknown OPS Code")
            else:
                parsed_data[field_name] = field_value

    # Handle the variable length of 'Emv Additional Data'
    # The 'm' length will be calculated based on the remaining bytes after reading the known fields
    m_length = len(packet_bytes) - pointer - 2  # Subtract 2 for ETX and LRC
    if m_length > 0:
        emv_data_field = packet_bytes[pointer : pointer + m_length]
        parsed_data["Emv Additional Data"] = emv_data_field.decode(
            "ascii", errors="ignore"
        ).rstrip()
        pointer += m_length

    # Return the parsed packet data
    return parsed_data


# Note: To use this function, you need to provide the actual packet byte string.
# The example_packet variable must be replaced with the actual data to parse.
# Here's how you would call the function:
# actual_packet_data = b'...'  # Your byte string here
# parsed_packet = parse_extended_packet(actual_packet_data)
# for key, value in parsed_packet.items():
#     print(f"{key}: {value}")

if __name__ == "__main__":
    example_packet = b"\x02925411880E0000000000040CLI0000001000004000000101011232232361Mastercard      ************1661   00000*                        ANTIQUARIUM CANNE BATTAGSP 142 CANNE DELLA BATTAARRIVEDERCI E GRAZIE    Cod. Commerc: 22537427                  N5A9KP493461400040002     00000000\x03\x01"
