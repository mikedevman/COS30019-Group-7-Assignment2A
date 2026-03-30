import math
from parser import parse_file
from yens import yens_k_shortest_paths, get_path_cost

def calculate_edge_cost(predicted_flow, km):
    # Converts flow to travel time in seconds

    # As given in Traffic Flow to Travel Time Conversion v1.0.pdf
    A = -1.4648375
    B = 93.75
    C = -predicted_flow
    discriminant = (B ** 2) - (4 * A * C)

    