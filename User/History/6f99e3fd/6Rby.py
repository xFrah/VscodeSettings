import math
import sys
import time
import numpy as np
import paho.mqtt.client as mqtt
import pygame
import asyncio

screen: pygame.surface.Surface
width: int
height: int


class Console:
    def __init__(self, window_size, portion=2 / 6):
        self.window_size = window_size
        width = window_size[0]
        height = window_size[1]
        self.portion = portion
        self.pos = (0, int(height * (1 - portion)))
        self.size = (width, int(height * portion))

        # create rectangle for console area to blit it later
        self.drawable_rect = pygame.Rect(self.pos, self.size)

    def get_drawable_rect(self):
        return self.drawable_rect

    def update_size(self, window_size):
        self.window_size = window_size
        width = window_size[0]
        height = window_size[1]
        self.pos = (0, int(height * (1 - self.portion)))
        self.size = (width, int(height * self.portion))

        # create rectangle for console area to blit it later
        self.drawable_rect = pygame.Rect(self.pos, self.size)


async def mqtt_loop():
    # create a client instance
    client = mqtt.Client()
    # connect to the broker
    client.connect("localhost", 1883, 60)
    # subscribe to the topic
    client.subscribe("test")

    # define a callback function
    def on_message(client, userdata, message):
        print("Message received: ", str(message.payload.decode("utf-8")))

    # set the callback function
    client.on_message = on_message

    # start the loop
    client.loop_start()
    # publish a message
    client.publish("test", "Hello World!")
    # stop the loop
    client.loop_stop()


def lidar_draw(lidar, distances, slices, people_angles, angle_lower_limit_1, angle_upper_limit_1, angle_lower_limit_2, angle_upper_limit_2):
    start = time.time()
    if distances and slices:
        # create a black image
        # img = np.zeros((1000, 1000, 3), np.uint8)

        # draw a circle in the center of the image
        # cv2.circle(img, (500, 500), 5, (255, 255, 255), -1)

        # draw the same circle but with pygame
        pygame.draw.circle(screen, (255, 255, 255), (width // 2, height // 2), 3)

        scale = 0.05

        # draw a line for each distance
        for angle, distance in distances:
            # convert distance and angle to x and y coordinates
            x = 500 + (distance * scale) * math.cos(math.radians(angle))
            y = 500 + (distance * scale) * math.sin(math.radians(angle))
            # draw a line from the center to the distance
            # cv2.circle(
            #    img,
            #    (int(x), int(y)),
            #    1,
            #    (0, 0, 255)
            #    if angle_lower_limit_1 <= angle <= angle_upper_limit_1 or angle_lower_limit_2 <= angle <= angle_upper_limit_2
            #    else (255, 0, 0),
            #    -1,
            # )

            # draw the same circle but with pygame
            pygame.draw.circle(
                screen,
                (0, 0, 255)
                if angle_lower_limit_1 <= angle <= angle_upper_limit_1 or angle_lower_limit_2 <= angle <= angle_upper_limit_2
                else (255, 0, 0),
                (int(x), int(y)),
                1,
            )

        # detection_frame = []

        for person_angle, _ in people_angles:
            distance, _ = lidar.get_distance_at_angle(slices, person_angle)
            # convert distance and angle to x and y coordinates
            x = 500 + (distance * scale) * math.cos(math.radians(person_angle))
            y = 500 + (distance * scale) * math.sin(math.radians(person_angle))
            # draw a line from the center to the distance
            # cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 5)

            # draw the same circle but with pygame
            pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), 5)
            # detection_frame.append((slice_index, rect, distance))

        # for slice_ in slices:
        #     medium_angle = sum([angle for angle, distance in slice_]) / len(slice_)
        #     medium_distance = min([distance for angle, distance in slice_ if distance > 30])
        #     # draw a line from the center to the distance
        #     cv2.line(img, (500, 500),
        #              (int(500 + (medium_distance * scale) * math.cos(math.radians(medium_angle))), int(500 + (medium_distance * scale) * math.sin(math.radians(medium_angle)))),
        #              (255, 255, 255), 1)

        # draw a purple circle at the angle

        # print(f"Detect >> Elapsed time for lidar: {(time.time() - start) * 1E3:.01f}ms")
        # cv2.imshow("Lidar", img)
    # print drawing time
    print(f"GUI >> Drawing time: {(time.time() - start) * 1E3:.01f}ms")


def pygame_loop():
    global screen, width, height

    pygame.init()
    pygame.display.set_caption("MQTT Display")
    # set screen as half the size of the display (for testing on a laptop)
    width, height = pygame.display.Info().current_w // 2, int(pygame.display.Info().current_h / 1.5)

    console = Console((width, height), 0.25)

    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

    # create a font
    font = pygame.freetype.SysFont("Cascadia Mono", 20)

    # create a text surface
    text = font.render("Logged!", (255, 255, 255))

    # main loop
    while True:
        # update width and height if windows has been resized
        if width != pygame.display.get_surface().get_width() or height != pygame.display.get_surface().get_height():
            width, height = pygame.display.get_surface().get_size()
            console.update_size((width, height))

        # fill the screen with black
        screen.fill((0, 0, 0))

        # draw the console area in dark gray color
        pygame.draw.rect(screen, (30, 30, 30), console.get_drawable_rect())

        # draw the text surface to the screen
        screen.blit(text[0], (10, console.pos[1] + 10))

        # update the screen
        pygame.display.update()

        # check for events
        for event in pygame.event.get():
            # check for quit event
            if event.type == pygame.QUIT:
                # quit pygame
                pygame.quit()
                # quit the program
                sys.exit()


# Run both the MQTT client and Pygame in the event loop
async def main():
    await asyncio.gather(mqtt_loop(), pygame_loop())


# Start the event loop
asyncio.run(main())


if __name__ == "__main__":
    main()
