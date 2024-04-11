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
    def __init__(self, name, x, y, radius):
        self.name = name
        self.x = x
        self.y = y
        self.radius = radius
        self.color = list(colors.values())[random.randint(0, len(colors) - 1)]

    def draw(self, img):
        cv2.circle(img, (self.x, self.y), self.radius, self.color, -1)
        cv2.putText(img, self.name, (self.x - 20, self.y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

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
traversed = {"WA"}

# porco dio e porca madonna che schifo di codice che ho scritto ma funziona quindi va bene così per ora (non funziona)


def check_violation(region, regions, constraints):
    for constraint in constraints[region]:
        if regions[region].color == regions[constraint].color:
            return True
    return False


def get_new_color(region, constraint, domains):
    new_color = list(domains[constraint])[random.randint(0, len(domains[constraint]) - 1)]
    print(f"[{region}] Changed {constraint} from {rgb_to_color[regions[constraint].color]} to {rgb_to_color[new_color]}")
    regions[constraint].change_color(new_color)
    return new_color


def constraint_propagation(region, regions, constraints, domains, traversed):
    print(f"Traversing {region} with domain {domains[region]}")

    # choose a random color from the domain
    color = list(domains[region])[random.randint(0, len(domains[region]) - 1)]
    print(f"[{region}] Set {region} to {rgb_to_color[color]}")
    regions[region].change_color(color)

    # remove the color from the domain of the neighbors
    for constraint in constraints[region]:
        domains[constraint].remove(color)
        print(f"[{region}] Removed {rgb_to_color[color]} from {constraint}")


asd = {region: list(constraints) for region, constraints in o_constraints.items()}
constraint_propagation("WA", regions, asd, domains, traversed)

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

print(list(o_constraints.items()))

if b:
    print("All constraints satisfied")


# create black image
img = np.zeros((512, 512, 3), np.uint8)
