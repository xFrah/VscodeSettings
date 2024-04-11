import random

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

constraints = {
    "WA": ["NT", "SA"],
    "NT": ["WA", "SA", "Q"],
    "SA": ["WA", "NT", "Q", "NSW", "V"],
    "Q": ["NT", "SA", "NSW"],
    "NSW": ["Q", "SA", "V"],
    "V": ["SA", "NSW", "T"],
    "T": ["V"],
}

region_colors = {r: list(colors.values())[random.randint(0, len(colors) - 1)] for r in list(constraints.keys())}

domains = {region: set(colors.values()) for region in region_colors}
traversed = set()


def constraint_propagation(region, region_colors, constraints, domains, traversed):
    print(f"Traversing {region} with domain {domains[region]}")
    traversed.add(region)

    # choose a random color from the domain
    color = list(domains[region])[random.randint(0, len(domains[region]) - 1)]
    print(f"[{region}] Set {region} to {rgb_to_color[color]}")
    region_colors[region] = color

    # remove the color from the domain of the neighbors
    for constraint in constraints[region]:
        if color in domains[constraint]:
            domains[constraint].remove(color)
        print(f"[{region}] Removed {rgb_to_color[color]} from {constraint}")

    # choose new neighbor to traverse
    for constraint in constraints[region]:
        if constraint not in traversed:
            constraint_propagation(constraint, region_colors, constraints, domains, traversed)


constraint_propagation("WA", region_colors, {region: list(constraints) for region, constraints in constraints.items()}, domains, traversed)

b = True

# check if all contraints are satisfied
for region, constraints in constraints.items():
    for constraint in constraints:
        if region_colors[region] == region_colors[constraint]:
            print(f"Constraint not satisfied: {region} and {constraint} have the same color")
            b = False
        else:
            print(
                f"Constraint satisfied: {region}, {constraint} = {rgb_to_color[region_colors[region]]}, {rgb_to_color[region_colors[constraint]]}"
            )

if b:
    print("All constraints satisfied")
