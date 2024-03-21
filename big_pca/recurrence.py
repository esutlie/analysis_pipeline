import matplotlib.pyplot as plt
import numpy as np


def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def recurrence(activity_list):
    """
    pick two trajectories
    distance plot between two with manhattan distance
    show distance matrix imshow
    linear regression on matrix as if it were scatter, weighted by the distances, where small distances are weighted much more heavily
    compute for every pair
    find the trajectory closest to the most others to use as a reference
    calculate lin regression on distance for each trajectory compared to that one
    do all that with leave one out

    """