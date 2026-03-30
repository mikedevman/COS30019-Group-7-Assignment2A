import os
import math
from parser import parse_file
from search_algorithms.yens import yens_k_shortest_paths, get_path_cost
import numpy as np
import joblib
import pickle
import random
from tensorflow.keras.models import load_model

def calculate_edge_cost(predicted_flow, distance_km):
    if predicted_flow <= 351:
        # Free flow conditions: maximum speed
        speed_kmh = 60.0
    else:
        # Congested flow: use quadratic model (A*v^2 + B*v + C = flow)
        A = -1.4648375
        B = 93.75
        C = -predicted_flow
        discriminant = (B ** 2) - (4 * A * C)
        
        if discriminant < 0:
            # Invalid speed calculation: use minimum safe speed
            speed_kmh = 32.0 
        else:
            # Solve quadratic and take the slower root
            speed_kmh = (-B - math.sqrt(discriminant)) / (2 * A)
            
        # Enforce speed bounds
        if speed_kmh > 60.0: 
            speed_kmh = 60.0
        elif speed_kmh < 10.0: 
            speed_kmh = 10.0

    # Calculate travel time: ensure non-zero distance and convert to seconds
    distance_not_zero = max(distance_km, 0.001)
    travel_time_seconds = (distance_not_zero / speed_kmh) * 3600  
    # Add 30-second intersection/node traversal penalty
    return travel_time_seconds + 30.0 

def update_graph_costs(edges, costs, nodes, model, scaler, X_test):
    dynamic_costs = {}
    edge_list = list(costs.keys())
    batch_size = len(edge_list)
    
    # Randomly sample sequences for prediction 
    random_indices = [random.randint(0, len(X_test) - 1) for _ in range(batch_size)]
    batch_sequences = np.array([X_test[i] for i in random_indices])
    
    # Get ML predictions for traffic flow on all edges
    raw_predictions = model.predict(batch_sequences, verbose=0) # verbose = 0 to make output cleaner
    # Convert normalized predictions back to vehicles/hour scale
    predicted_flows = scaler.inverse_transform(raw_predictions)
    
    # Calculate new travel times for each edge
    for i, (node_a, node_b) in enumerate(edge_list):
        predicted_flow = predicted_flows[i][0]
        origin_distance_km = costs[(node_a, node_b)]
        # Convert distance degrees to kilometers - 111 km per degree is the approximate conversion at the equator
        dist_km = origin_distance_km * 111.0 
        # Calculate cost based on predicted flow
        new_time_seconds = calculate_edge_cost(predicted_flow, dist_km)
        dynamic_costs[(node_a, node_b)] = new_time_seconds
        
    return dynamic_costs

def run_tbrgs(filename, origin, destination, model_name='lstm'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    model_path = os.path.join(base_dir, 'models', model_name, f'{model_name}_traffic_model.keras')
    scaler_path = os.path.join(base_dir, 'models', model_name, f'{model_name}_scaler.pkl')
    data_path = os.path.join(base_dir, 'data', 'preprocessed', f'preprocessed_data_{model_name}.pkl')
    
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'models', 'lstm', 'lstm_traffic_model.keras')
        scaler_path = os.path.join(base_dir, 'models', 'lstm', 'lstm_scaler.pkl')
        data_path = os.path.join(base_dir, 'data', 'preprocessed', 'preprocessed_data_lstm.pkl')

    # Parse the map file to extract graph structure and static costs
    nodes, edges, static_costs, _, _ = parse_file(filename)

    # Load trained model and preprocessing objects
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(data_path, "rb") as f:
        X_test = pickle.load(f)["X_test"]
    
    # Update edge costs based on predicted traffic flow
    dynamic_costs = update_graph_costs(edges, static_costs, nodes, model, scaler, X_test)
    
    # Find K=5 shortest paths from origin to destination
    top_paths = yens_k_shortest_paths(origin, [destination], edges, dynamic_costs, nodes, K=5)

    # Format results for output
    results = []
    for i, path in enumerate(top_paths):
        # Get total cost for this route and convert to minutes
        cost_seconds = get_path_cost(path, dynamic_costs)
        cost_mins = cost_seconds / 60
        results.append({
            "route": i + 1,
            "estimated_time_mins": round(cost_mins, 1),
            "path": path
        })
        
    return results

if __name__ == "__main__":
    # Test the TBRGS system by finding routes from zone 7001 to zone 7015
    routes = run_tbrgs(filename="Assignment2B/map_data/boroodara_tbrgs_map_coordinates.txt", origin="7001", destination="7015")
    
    # Display all recommended routes with estimated travel times and paths
    for r in routes:
        print(f"Route {r['route']} | Est. Time: {r['estimated_time_mins']} mins | Path: {' -> '.join(map(str, r['path']))}")
