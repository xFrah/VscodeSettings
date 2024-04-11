import random
import cv2
import numpy as np

colors = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
}

rgb_to_color = {
    (255, 0, 0): "blue",
    (0, 255, 0): "green",
    (0, 0, 255): "red",
}


class Region:
    def __init__(self,):
        self.name = name
        self.color = list(colors.values())[random.randint(0, len(colors) - 1)]

    def change_color(self, color):
        self.color = color


regions = {
    "WA": Region("WA", 100, 100, 50),
    "NT": Region("NT", 200, 100, 50),
    "SA": Region("SA", 300, 100, 50),
    "Q": Region("Q", 100, 200, 50),
    "NSW": Region("NSW", 200, 200, 50),
    "V": Region("V", 300, 200, 50),
    "T": Region("T", 200, 300, 50),
}

o_constraints = {
    "WA": ["NT", "SA"],
    "NT": ["WA", "SA", "Q"],
    "SA": ["WA", "NT", "Q", "NSW", "V"],
    "Q": ["NT", "SA", "NSW"],
    "NSW": ["Q", "SA", "V"],
    "V": ["SA", "NSW", "T"],
    "T": ["V"],
}

domains = {region: set(colors.values()) for region in regions}
traversed = set()


def constraint_propagation(region, regions, constraints, domains, traversed):
    print(f"Traversing {region} with domain {domains[region]}")
    traversed.add(region)

    # choose a random color from the domain
    color = list(domains[region])[random.randint(0, len(domains[region]) - 1)]
    print(f"[{region}] Set {region} to {rgb_to_color[color]}")
    regions[region].change_color(color)

    # remove the color from the domain of the neighbors
    for constraint in constraints[region]:
        if color in domains[constraint]:
            domains[constraint].remove(color)
        print(f"[{region}] Removed {rgb_to_color[color]} from {constraint}")

    # choose new neighbor to traverse
    for constraint in constraints[region]:
        if constraint not in traversed:
            constraint_propagation(constraint, regions, constraints, domains, traversed)


constraint_propagation("WA", regions, {region: list(constraints) for region, constraints in o_constraints.items()}, domains, traversed)

b = True

# check if all contraints are satisfied
for region, constraints in o_constraints.items():
    for constraint in constraints:
        if regions[region].color == regions[constraint].color:
            print(f"Constraint not satisfied: {region} and {constraint} have the same color")
            b = False
        else:
            print(
                f"Constraint satisfied: {region}, {constraint} = {rgb_to_color[regions[region].color]}, {rgb_to_color[regions[constraint].color]}"
            )

if b:
    print("All constraints satisfied")


# create black image
img = np.zeros((512, 512, 3), np.uint8)
