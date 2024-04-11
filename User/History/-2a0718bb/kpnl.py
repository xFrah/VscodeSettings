from land import FootballField
import webbrowser
import folium
import numpy as np
import cv2
from helpers import find_shortest_path

coords = [(41.934481, 12.455002), (41.934329, 12.454218), (41.933425, 12.454537), (41.933574, 12.455324)]
coords = [(y, x) for x, y in coords]

field = FootballField(coords)
print(field)

h = 500

# open football field color cv2 image and plot it, with callback. it must have h height and 1.67h width.
image = np.zeros((h, int(h * 1.67), 3), np.uint8)
image[:] = (0, 140, 20)
markers = []


# mouse callback function
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        x_frac = x / image.shape[1]
        y_frac = 1 - (y / image.shape[0])
        markers.append((x_frac, y_frac))
        cv2.circle(image, (x, y), 30, (255, 0, 0), -1)


# create cv2 window with click callback
cv2.namedWindow("Football field")
cv2.setMouseCallback("Football field", draw_circle)

edges = []


while True:
    for edge in edges:
        cv2.line(image, edge[0], edge[1], (0, 0, 0), 3)
    cv2.imshow("Football field", image)
    k = cv2.waitKey(1) & 0xFF
    # if enter is detected
    if k == 13:
        print(field.gdf.crs)

        # plot contour from vertices
        fmap = field.gdf.explore()
        print(markers)
        markers_frac = [field.frac_to_coord(*marker)[::-1] for marker in markers]
        permutation, distance = find_shortest_path(markers_frac)
        print(permutation)
        print(f"Distance traveled: {distance * 111000:.2f}m")

        for i, marker in enumerate(markers):
            coords = field.frac_to_coord(*marker)[::-1]
            folium.Marker(coords, popup=f"Waypoint {i}").add_to(fmap)

        # create a folium polyline from the shortest path
        folium.PolyLine([field.frac_to_coord(*markers_frac[i])[::-1] for i in permutation], color="red").add_to(fmap)

        outfp = r"map.html"

        fmap.save(outfp)

        webbrowser.open(outfp)
