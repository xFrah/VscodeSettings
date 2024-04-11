import matplotlib.pyplot as plt
import numpy


def compute_maximum_speed(max_deceleration, current_speed, distance):
    max_deceleration = abs(max_deceleration)  # Make sure deceleration is positive

    if distance <= 0:
        return 0  # No distance to cover, so no speed is required

    time_to_stop = abs(current_speed / max_deceleration)  # Time required to stop
    stopping_distance = max(0, 0.5 * max_deceleration * time_to_stop**2)  # Distance covered while stopping

    remaining_distance = distance - stopping_distance
    max_speed = (2 * max_deceleration * remaining_distance + current_speed**2) ** 0.5

    return max_speed


def compute_correlation_coefficient(last_speeds):
    if numpy.std(last_speeds) == 0:
        return 0
    else:
        return numpy.corrcoef(last_speeds, range(len(last_speeds)))[0, 1]


# simulate the maximum speed of the car as a function of the distance to the next car
distance = 1500  # distance to the next car (m)
actual_speed = 50  # current speed of the car (m/s)
deceleration = 4.572  # maximum deceleration of the car (m/s^2)

speeds = []
max_speeds = []
alpha = 0.01  # the maximum speed is 90% of the maximum speed
alpha2 = 0.4

last_speeds = []
n = 5

# each iteration is 0.1 second
while True:
    # compute new distance based on the fact that 1 iteration is 0.1 second
    distance = distance - actual_speed

    # compute the maximum speed of the car
    max_speed = compute_maximum_speed(deceleration, actual_speed, distance)
    print("speed: " + str(actual_speed) + " max_speed: " + str(max_speed), " distance: " + str(distance))
    if actual_speed > max_speed:
        if len(last_speeds) == n:
            alpha = abs(compute_correlation_coefficient(last_speeds)) * alpha2 + (1 - alpha2) * alpha
            if alpha == 0:
                alpha = 0.01
            print("alpha: " + str(alpha), " last_speeds: " + str(last_speeds))
            actual_speed = (1 - alpha) * actual_speed + alpha * max_speed
        else:
            print("Still not enough data to compute alpha")

    last_speeds.append(actual_speed)
    if len(last_speeds) > n:
        last_speeds.pop(0)
    speeds.append(actual_speed)
    max_speeds.append(max_speed)
    if distance <= 0:
        break

    # plot speeds
    plt.clf()
    plt.plot(speeds)
    plt.plot(max_speeds, color="red", linestyle="dashed")
    plt.ylabel("speed (m/s)")
    plt.xlabel("time (s)")
    print("Correlation coefficient: " + str(compute_correlation_coefficient([1, 3, 2, 1])))
    
    # turn into image with opencv 
    # plt.show()
