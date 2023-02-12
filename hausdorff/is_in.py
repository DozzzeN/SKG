import math
import sys


def is_in_poly(p, polygon):
    # https://blog.csdn.net/leviopku/article/details/111224539
    """
    :param p: [x, y]
    :param polygon: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(polygon):
        next_i = i + 1 if i + 1 < len(polygon) else 0
        x1, y1 = corner
        x2, y2 = polygon[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


from shapely.geometry import Polygon


# 平均单向hausdorff距离的效果差些
def average_hd(x, y):
    h = 0
    for xi in x:
        shortest = sys.maxsize
        for yi in y:
            d = round(math.pow(xi[0] - yi[0], 2) + math.pow(xi[1] - yi[1], 2), 10)
            if d < shortest:
                shortest = d
        h += shortest
    return h / len(x)


if __name__ == '__main__':
    print(Polygon([(0, 0), (1, 1), (1, 0)]).intersects(Polygon([(0, 0), (1, 0), (0, 1), (-1, 0)])))

    point = [0, 1 / 2]
    poly = [[0, 0], [1, 0], [1, 1], [0, 1]]
    print(is_in_poly(point, poly))

    p1 = Polygon([(0, 0), (1, 0), (0, 1), (1, 1)])
    print(p1.convex_hull)

    p2 = Polygon([(0, 0), (0, 5), (2, 2), (3, 3), (5, 0)])
    p3 = p2.convex_hull
    print(list(p2.convex_hull.exterior.coords))

    print(average_hd([[0, 0], [1, 0], [1, 1], [0, 1]], [(0, 0), (0, 5), (2, 2), (3, 3), (5, 0)]))
