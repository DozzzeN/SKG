from pymobility.models.mobility import random_waypoint

rw = random_waypoint(200, dimensions=(100, 100), velocity=(0.1, 1.0), wt_max=1.0)
for positions in rw:
    print(positions)
