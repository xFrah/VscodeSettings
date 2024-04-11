import serial
import struct
import time

# Packet formats
serial_act_t4_in_format = "<B3h2fB"  # 1 int8_t, 3 int16_t, 2 float, 1 uint8_t

# Open serial port
ser = serial.Serial("COM7", 921600)  # Replace 'COM_PORT' with your COM port
total_degrees = 0, 0, 0


s = 0x9A


def send_packet(packet_data, packet_format):
    packed_data = struct.pack(packet_format, *packet_data)
    print("Sending:", packed_data)
    ser.write(s)
    ser.write(packed_data)


def move_servos(speed, duration):
    # Send command to move servos at given speed
    in_packet = [
        1,
        speed,
        speed,
        speed,
        0.0,
        0,
        255,
    ]  # Placeholder values for non-speed fields
    checksum = 0
    for i in range(0, len(in_packet) - 1):
        checksum += in_packet[i]
    in_packet[-1] = int(checksum % 256)
    in_packet = tuple(in_packet)
    send_packet(in_packet, serial_act_t4_in_format)

    # Wait for the duration
    time.sleep(duration)

    # Send command to move servos at opposite speed
    in_packet = (1, -speed, -speed, -speed, 0.0, 0, 255)
    send_packet(in_packet, serial_act_t4_in_format)

    # Wait for the same duration
    time.sleep(duration)


try:
    # Example usage
    move_servos(
        -3000, 5
    )  # Move servos at speed 5000 for 5 seconds, then -5000 for 5 seconds
except KeyboardInterrupt:
    print("Operation cancelled.")


def receive_packet(packet_format):
    data_size = struct.calcsize(packet_format)
    received_data = ser.read(data_size)
    return struct.unpack(packet_format, received_data)


i = 0
while True:
    i += 1
    if i % 100 == 0:
        print("Sending packet")
        move_servos(-1000, 5)
    # receive serial stream
    l = ser.readline()
    print(l)
    time.sleep(0.1)
