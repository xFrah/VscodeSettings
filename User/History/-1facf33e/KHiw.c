#include <stdio.h>
#include <stdlib.h>

#define WIDTH 700
#define HEIGHT 600
#define num_particles 100
#define particle_radius 5
#define seed 17

struct Particle {
    int id;
    int x;
    int y;
    char walker;
    int y_index;
};

struct Particle particles[num_particles];
struct Particle *particles_x[num_particles];
struct Particle *particles_y[num_particles];

void init_particles() {
    srand(seed);
    for (int i = 0; i < num_particles; i++) {
        particles[i].id = i;
        particles[i].x = rand() % WIDTH;
        particles[i].y = rand() % HEIGHT;
        particles[i].walker = rand() % 2;
        particles[i].y_index = i;
        particles_x[i] = &particles[i];
        particles_y[i] = &particles[i];
    }
}

void move_particles() {
    for (int i = 0; i < num_particles; i++) {
        if (particles[i].walker) {
            particles[i].x = particles[i].x + 1 + (-2 * (rand() % 2));
        } else {
            particles[i].y = particles[i].y + 1 + (-2 * (rand() % 2));
        }
    }
}

void set_particle_y_index() {
    for (int i = 0; i < num_particles; i++) {
        particles_y[i]->y_index = i;
    }
}


int main() {
    init_particles();
    while (1) {
        // sort
        set_particle_y_index();
        move_particles();
        for (int i = 0; i < num_particles - 1; i++) {
            printf("Particle %d: x=%d, y=%d\n", particles[i].id, particles[i].x, particles[i].y);
        }
    }
    return 0;
}