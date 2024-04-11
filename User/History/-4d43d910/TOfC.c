#include <asm-generic/socket.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#define PORT 6789
#define num_particles 1000

struct Particle {
    int id;
    int x;
    int y;
    char walker;
    int y_index;
};

// make a struct that contains header, packet length, packet data and footer
struct Packet {
    char header[4];
    int length;
    struct Particle particles[num_particles];
    char footer[4];
};

int socket_server_start() {
    // ssize_t valread;

    printf("Starting server\n");

    // Creating socket file descriptor
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in address;
    int new_socket;
    socklen_t addrlen = sizeof(address);
    int opt = 1;

    printf("Socket created\n");

    // Forcefully attaching socket to the port 8080
    if (setsockopt(server_fd, SOL_SOCKET,
                   SO_REUSEADDR | SO_REUSEPORT, &opt,
                   sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    printf("Socket options set\n");

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    printf("Address set\n");

    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr *)&address,
             sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    printf("Socket bound\n");
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    printf("Listening\n");
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                             &addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    printf("Connection accepted\n");
    // valread = read(new_socket, buffer,
    //                1024 - 1); // subtract 1 for the null
    //                           // terminator at the end

    // closing the connected socket
    // close(new_socket);
    // closing the listening socket
    // close(server_fd);
    return new_socket;
}

void socket_server_send(int socket, void *particles, int length) {
    int nbs = send(socket, particles, length, 0);
    if (nbs < 0) {
        perror("send");
        exit(EXIT_FAILURE);
    }
}