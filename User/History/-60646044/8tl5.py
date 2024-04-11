import pygame
import sys
import random
import socket
import struct
import threading

current_packet = {}

class Particle:
    def __init__(self, iid, x, y, walker, y_index):
        self.id = iid
        self.x = x
        self.y = y
        self.walker = walker
        self.y_index = y_index

# Define the packet and particle structure formats
header_format = '4s'  # 4 characters for the header
length_format = 'i'   # integer for the length
footer_format = '4s'  # 4 characters for the footer
particle_format = '=iiiixi'

# Calculate the sizes of each part
header_size = struct.calcsize(header_format)
length_size = struct.calcsize(length_format)
footer_size = struct.calcsize(footer_format)
particle_size = struct.calcsize(particle_format)

def receive_and_parse_packets(sock):
    global current_packet
    buffer = b''
    while True:
        # Receive data from the socket
        data = sock.recv(50000)
        if not data:
            print("Connection closed by the server.")
            break

        buffer += data

        # Process complete packets from the buffer
        while True:
            # Search for packet header
            start_idx = buffer.find(b'PART')
            if start_idx == -1:
                # Header not found, break and wait for more data
                break

            # Check if we have enough data to read the length
            if len(buffer) < start_idx + header_size + length_size:
                break  # Wait for more data

            # Extract the length of the particle data
            length_offset = start_idx + header_size
            (length,) = struct.unpack(length_format, buffer[length_offset:length_offset + length_size])

            # Check if we have the full packet
            packet_end = length_offset + length_size + length + footer_size
            if len(buffer) < packet_end:
                break  # Wait for more data

            # Verify the footer
            footer = buffer[packet_end - footer_size:packet_end]
            if footer != b'ENDP':
                print("Invalid footer. Packet may be corrupted.")
                break

            particles_data = buffer[length_offset + length_size:packet_end - footer_size]

            # Dictionary comprehension to create a dictionary of Particle objects
            current_packet = {
                struct.unpack(particle_format, particles_data[i:i + particle_size])[0]: Particle(*struct.unpack(particle_format, particles_data[i:i + particle_size]))
                for i in range(0, length, particle_size)
            }

            # Remove the processed packet from the buffer
            buffer = buffer[packet_end:]

# Constants
WIDTH = 800  # in pixels
HEIGHT = 600
gray = (200, 200, 200)
black = (0, 0, 0)
dark_red = (128, 0, 0)
yellow = (255, 255, 0)

# pygame window that can display particle
def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 6789)
    print(f"Connecting to {server_address[0]} port {server_address[1]}")
    sock.connect(server_address)
    thread = threading.Thread(target=receive_and_parse_packets, args=(sock,), daemon=True)
    thread.start()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Random Walk")

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(black)

        for _, particle in current_packet.items():
            pygame.draw.circle(
                screen,
                gray if particle.walker else dark_red,
                (particle.x, particle.y),
                5,
                5,
            )

        pygame.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()
