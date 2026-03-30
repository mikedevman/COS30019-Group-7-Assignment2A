import math
from parser import parse_file
from yens import yens_k_shortest_paths, get_path_cost

def calculate_edge_cost(predicted_flow, distance_km):
    # Converts flow to travel time in seconds

    # As given in Traffic Flow to Travel Time Conversion v1.0.pdf
    A = -1.4648375
    B = 93.75
    C = -predicted_flow
    discriminant = (B ** 2) - (4 * A * C)

    if discriminant < 0:
        speed_kmh = 5.0
    else:
        speed_kmh = (-B + math.sqrt(discriminant)) / (2 * A)

    if speed_kmh > 60.0: 
        speed_kmh = 60.0

    travel_time_seconds = (distance_km / speed_kmh) * 3600  
    return travel_time_seconds + 30.0 # add 30s intersection delay

def update_graph_costs(edges, costs, nodes):
    # Updates the static distances with dynamic ML travel times
    dynamic_costs = {}

    model = load_model()

    for (node_a, node_b), original_distance_km in costs.items():
        predicted_flow = model.predict()
        distance_km = original_distance_km * 111.0 # one degree of latitude
        new_cost = calculate_edge_cost(predicted_flow, distance_km)
        dynamic_costs[(node_a, node_b)] = new_cost
    return dynamic_costs

def run_tbrgs(filename, origin, destination):
    print(f"Loading map: {filename}...")
    nodes, edges, static_costs, _, _ = parse_file(filename)
    
    print("Updating edge costs with ML traffic predictions...")
    dynamic_costs = update_graph_costs(edges, static_costs, nodes)
    
    print(f"Calculating Top 5 routes from {origin} to {destination}...\n")
    top_paths = yens_k_shortest_paths(origin, [destination], edges, dynamic_costs, nodes, K=5)

    for i, path in enumerate(top_paths):
        cost_seconds = get_path_cost(path, dynamic_costs)
        cost_mins = cost_seconds / 60
        print(f"Route {i + 1} | Est. Time: {cost_mins:.1f} mins | Path: {' -> '.join(map(str, path))}")

if __name__ == "__main__":
    run_tbrgs()


