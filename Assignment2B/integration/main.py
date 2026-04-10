import os
import sys
import math
from parser import parse_file
from search_algorithms.yens import yens_k_shortest_paths, get_path_cost
import numpy as np
import joblib
import pickle
from tensorflow.keras.models import load_model

# Import GCN-LSTM nn.Module classes from the notebooks directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
_gcn_core_path = os.path.abspath(os.path.join(_current_dir, '..', 'notebooks', 'gcn_lstm_core'))
if _gcn_core_path not in sys.path:
    sys.path.insert(0, _gcn_core_path)

# Conditional imports for PyTorch (only when needed)
def _import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, torch.nn
    except ImportError:
        raise ImportError("PyTorch is required for GCN-LSTM model. Please install it with: pip install torch")

def _load_gcn_lstm_class():
    """Import the proper nn.Module GCN_LSTM from gcn_lstm_classes so that
    load_state_dict correctly restores all nested layer weights."""
    try:
        from gcn_lstm_classes import GCN_LSTM
        return GCN_LSTM
    except ImportError as e:
        raise ImportError(
            f"Could not import GCN_LSTM from gcn_lstm_classes.py: {e}\n"
            f"Expected location: {_gcn_core_path}"
        )

def calculate_edge_cost(predicted_flow, distance_km, speed_limit_kmh=60.0, intersection_delay_s=30.0):
    if predicted_flow <= 351:
        # Free flow conditions: maximum speed
        speed_kmh = speed_limit_kmh
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
        if speed_kmh > speed_limit_kmh: 
            speed_kmh = speed_limit_kmh
        elif speed_kmh < 10.0: 
            speed_kmh = 10.0

    # Calculate travel time: ensure non-zero distance and convert to seconds
    distance_not_zero = max(distance_km, 0.001)
    travel_time_seconds = (distance_not_zero / speed_kmh) * 3600  
    # Add intersection/node traversal penalty
    return travel_time_seconds + intersection_delay_s

def _find_hour_indices(X_test, depart_hour, model_type, feature_cols=None):
    """
    Return an array of X_test indices whose sequences correspond to the requested
    departure hour. The last time step of each sequence is the 15-min interval
    immediately before the predicted step, so we match on that hour value.

    Supports two feature layouts:
      - LSTM/GRU  : shape (N, seq_len, features), hour_of_day at fixed index 2,
                    MinMaxScaled from range 0-23 -> 0-1.
      - GCN-LSTM  : shape (N, seq_len, nodes, features), hour_of_day or hour_sin/
                    hour_cos located by name in feature_cols, averaged across nodes.

    Falls back to all indices when no matching sequences are found.
    """
    KERAS_HOUR_IDX = 2      # position in ['Traffic_Volume','day_of_week','hour_of_day',...]
    HOUR_SCALE     = 23.0   # MinMaxScaler maps 0-23 hours onto 0-1
    TOLERANCE_H    = 1      # +-1 hour window around requested time

    try:
        if model_type == 'gcn_lstm' and feature_cols is not None:
            feature_cols = list(feature_cols)

            if 'hour_of_day' in feature_cols:
                # MinMaxScaled hour_of_day — average across nodes at last timestep
                h_idx = feature_cols.index('hour_of_day')
                last_hour_scaled = X_test[:, -1, :, h_idx].mean(axis=1)
                last_hour = np.round(last_hour_scaled * HOUR_SCALE).astype(int)

            elif 'hour_sin' in feature_cols and 'hour_cos' in feature_cols:
                # Cyclical encoding — reconstruct hour from atan2, average across nodes
                sin_idx = feature_cols.index('hour_sin')
                cos_idx = feature_cols.index('hour_cos')
                sin_vals = X_test[:, -1, :, sin_idx].mean(axis=1)
                cos_vals = X_test[:, -1, :, cos_idx].mean(axis=1)
                angles   = np.arctan2(sin_vals, cos_vals) % (2 * np.pi)
                last_hour = np.round(angles / (2 * np.pi) * 24).astype(int) % 24

            else:
                # No recognisable hour feature — cannot filter
                return np.arange(len(X_test))

        else:
            # LSTM / GRU: hour_of_day is always at feature index 2, scaled 0->1
            last_hour_scaled = X_test[:, -1, KERAS_HOUR_IDX]
            last_hour = np.round(last_hour_scaled * HOUR_SCALE).astype(int)

        # Build a boolean mask allowing +-TOLERANCE_H hours (wraps midnight)
        diff     = np.abs(last_hour - depart_hour)
        diff     = np.minimum(diff, 24 - diff)      # handle midnight wrap
        matching = np.where(diff <= TOLERANCE_H)[0]

        return matching if len(matching) > 0 else np.arange(len(X_test))

    except Exception:
        # Any unexpected indexing error — fall back to using all sequences
        return np.arange(len(X_test))

def update_graph_costs(edges, costs, nodes, model, scaler, X_test, model_type='keras', A=None, node_to_idx=None, speed_limit_kmh=60.0, intersection_delay_s=30.0, depart_hour=8, feature_cols=None):
    dynamic_costs = {}
    edge_list = list(costs.keys())
    batch_size = len(edge_list)
    
    if model_type == 'gcn_lstm':
        # Import torch locally
        torch, _ = _import_torch()
        # GCN-LSTM predicts for all nodes simultaneously
        # Select sequences that match the requested departure hour
        candidate_indices = _find_hour_indices(X_test, depart_hour, model_type, feature_cols)
        sampled_indices   = np.random.choice(candidate_indices, size=batch_size, replace=True)
        batch_sequences   = np.array([X_test[i] for i in sampled_indices])
        
        # Convert to torch tensor and add batch dimension if needed
        batch_sequences = torch.tensor(batch_sequences, dtype=torch.float32)
        if len(batch_sequences.shape) == 3:  # (batch, time, nodes, features) expected
            batch_sequences = batch_sequences.unsqueeze(0)  # Add batch dimension
        
        model.eval()
        with torch.no_grad():
            raw_predictions = model(batch_sequences).cpu().numpy()  # (batch, nodes)
        
        # Convert normalized predictions back to vehicles/hour scale
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ > 1:
            # Multi-feature scaler (e.g. 13 features), target (Traffic_Volume) is the first feature
            # Use formula: X_orig = X_scaled * (max - min) + min
            min_val = scaler.data_min_[0]
            max_val = scaler.data_max_[0]
            predicted_flows_all_nodes = raw_predictions * (max_val - min_val) + min_val
        else:
            predicted_flows_all_nodes = scaler.inverse_transform(raw_predictions)
        
        # Map node predictions to edge predictions (use source node traffic volume)
        predicted_flows = []
        for node_a, node_b in edge_list:
            if node_a in node_to_idx:
                node_idx = node_to_idx[node_a]
                predicted_flow = predicted_flows_all_nodes[0, node_idx]  # Take first batch
            else:
                predicted_flow = 100.0  # Default moderate flow
            predicted_flows.append([predicted_flow])
        predicted_flows = np.array(predicted_flows)
    else:
        # Keras models (LSTM, GRU) - predict per edge
        # Select sequences that match the requested departure hour
        candidate_indices = _find_hour_indices(X_test, depart_hour, model_type)
        sampled_indices   = np.random.choice(candidate_indices, size=batch_size, replace=True)
        batch_sequences   = np.array([X_test[i] for i in sampled_indices])
        
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
        new_time_seconds = calculate_edge_cost(predicted_flow, dist_km, speed_limit_kmh, intersection_delay_s)
        dynamic_costs[(node_a, node_b)] = new_time_seconds
        
    return dynamic_costs

def get_path_distance_km(path, static_costs):
    total_degrees = 0
    for i in range(len(path) - 1): # - 1 to avoid index out of bounds
        total_degrees += static_costs.get((path[i], path[i+1]), 0)
    return total_degrees * 111.0

def run_tbrgs(filename, origin, destination, model_name, k_routes=5, speed_limit=60.0, intersection_delay=30.0, depart_time='08:00'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    if model_name == 'lstm':
        model_dir = 'lstm'
        model_file = 'lstm_model.keras'
        scaler_file = 'lstm_scaler.pkl'
        data_file = 'preprocessed_data_lstm.pkl'
        model_type = 'keras'
    elif model_name == 'bidirectional_lstm':
        model_dir = 'lstm'
        model_file = 'lstm_bidirectional_model.keras'
        scaler_file = 'lstm_scaler.pkl'
        data_file = 'preprocessed_data_lstm.pkl'
        model_type = 'keras'
    elif model_name == 'gru':
        model_dir = 'gru'
        model_file = 'gru_model.keras'
        scaler_file = 'gru_scaler.pkl'
        data_file = 'preprocessed_data_gru.pkl'
        model_type = 'keras'
    elif model_name == 'bidirectional_gru':
        model_dir = 'gru'
        model_file = 'gru_bidirectional_model.keras'
        scaler_file = 'gru_scaler.pkl'
        data_file = 'preprocessed_data_gru.pkl'
        model_type = 'keras'
    elif model_name == 'custom_gcn_lstm':
        model_dir = 'custom_gcn_lstm'
        model_file = 'custom_gcn_lstm_model.pth'
        scaler_file = 'custom_gcn_lstm_scaler.pkl'
        data_file = 'preprocessed_data_custom_gcn_lstm.pkl'
        model_type = 'gcn_lstm'
    else:
        model_dir = model_name
        model_file = f'{model_name}_traffic_model.keras'
        scaler_file = f'{model_name}_scaler.pkl'
        data_file = f'preprocessed_data_{model_name}.pkl'
        model_type = 'keras'
    
    model_path = os.path.join(base_dir, 'models', model_dir, model_file)
    scaler_path = os.path.join(base_dir, 'models', model_dir, scaler_file)
    data_path = os.path.join(base_dir, 'data', 'preprocessed', data_file)

    # Parse departure hour from "HH:MM" string for time-based sequence selection
    try:
        depart_hour = int(depart_time.split(':')[0])
        depart_hour = max(0, min(23, depart_hour))  # clamp to valid range
    except (ValueError, AttributeError, IndexError):
        depart_hour = 8  # default to morning peak if parsing fails

    # Parse the map file to extract graph structure and static costs
    nodes, edges, static_costs, _, _ = parse_file(filename)

    # Load trained model and preprocessing objects
    # feature_cols is only available for GCN-LSTM; stays None for Keras models
    feature_cols = None
    if model_type == 'gcn_lstm':
        # Load PyTorch model using the proper nn.Module class so that
        # load_state_dict correctly restores all nested layer weights
        torch, _ = _import_torch()
        GCN_LSTM = _load_gcn_lstm_class()

        with open(data_path, "rb") as f:
            gcn_data = pickle.load(f)
        A = gcn_data["A"]
        node_to_idx = gcn_data["node_to_idx"]
        num_nodes = len(node_to_idx)
        feature_cols = gcn_data["feature_cols"]
        in_features = len(feature_cols)

        model = GCN_LSTM(A, num_nodes, in_features)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)  # nn.Module.load_state_dict restores all nested weights correctly
        model.eval()
        scaler = joblib.load(scaler_path)
        X_test = gcn_data["X_test"]
    else:
        # Load Keras model
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        # Handle different scaler formats: for GRU it's a dict, for LSTM it's the scaler object
        if isinstance(scaler, dict):
            scaler = scaler["scaler_y"]
        with open(data_path, "rb") as f:
            X_test = pickle.load(f)["X_test"]

    # Update edge costs using ML-predicted traffic flow for the requested departure hour
    dynamic_costs = update_graph_costs(
        edges, static_costs, nodes, model, scaler, X_test,
        model_type,
        A           if model_type == 'gcn_lstm' else None,
        node_to_idx if model_type == 'gcn_lstm' else None,
        speed_limit, intersection_delay,
        depart_hour, feature_cols
    )
    
    # Find K shortest paths from origin to destination
    top_paths = yens_k_shortest_paths(origin, [destination], edges, dynamic_costs, nodes, K=k_routes)

    # Format results for output
    results = []
    for i, path in enumerate(top_paths):
        # Get total cost for this route and convert to minutes
        cost_seconds = get_path_cost(path, dynamic_costs)
        cost_mins = cost_seconds / 60
        distance_km = get_path_distance_km(path, static_costs)
        
        results.append({
            "route": i + 1,
            "estimated_time_mins": round(cost_mins, 1),
            "distance_km": round(distance_km, 1),
            "path": path
        })
        
    return results