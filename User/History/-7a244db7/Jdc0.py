

class ROI:
    def __init__(self, roi_path):
        self.roi_polygon = []
        with open(roi_path, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                self.roi_polygon.append((int(row[0]), int(row[1])))

    def is_inside(self, x, y):
        n = len(self.roi_polygon)
        inside = False
        p1x, p1y = self.roi_polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.roi_polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
