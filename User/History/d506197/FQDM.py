import geopandas as gpd
from typing import List, Tuple, Union
import pandas as pd
from shapely.geometry import Point
import numpy as np
import cv2
from helpers import order_points


class Land:
    def __init__(self, vertices: List[Point]) -> None:
        self.vertices: List[Point] = vertices
        self.vertices_tuples: List[Tuple[float, float]] = [(point.x, point.y) for point in vertices]
        self.gdf = gpd.GeoDataFrame(pd.DataFrame({"Corner": list(range(len(vertices))), "geometry": vertices}))
        self.gdf.set_crs(epsg=4326, inplace=True)

    def __str__(self) -> str:
        return str(self.gdf)


class FootballField(Land):
    FOOTBALL_FIELD_RATIO = 1.67

    def __init__(self, vertices: List[Tuple[float, float]]) -> None:
        """
        Given a list of vertices, create a football field, which approximates a rectangle.
        """
        # TODO order points
        vertices = order_points(vertices)
        self.
        super().__init__([Point(v) for v in vertices])

    def get_points(self, as_tuples=False) -> Union[List[Point], List[Tuple[float, float]]]:
        """
        Return the vertices of the football field.
        """
        return self.vertices_tuples if as_tuples else self.vertices

    def frac_to_coord(self, x, y) -> Tuple[float, float]:
        """
        Given a fraction over the field, return the real-world coordinate.
        """
        # Normalized coordinates
        src_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")

        # Real-world coordinates
        dst_pts = np.array(self.vertices_tuples, dtype="float32")

        # Compute the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Map the normalized point to real-world point
        point = np.array([[x, y]], dtype="float32")
        real_point = cv2.perspectiveTransform(point[None, :, :], matrix)

        return real_point[0][0]
