import os
import math
from parser import parse_file
from search_algorithms.yens import yens_k_shortest_paths, get_path_cost
import numpy as np
import joblib
import pickle
import random
from tensorflow.keras.models import load_model

# Conditional imports for PyTorch (only when needed)
def _import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, torch.nn
    except ImportError:
        raise ImportError("PyTorch is required for GCN-LSTM model. Please install it with: pip install torch")

# =========================
# GCN LAYER (D^{-1/2} A_tilde D^{-1/2})
# =========================
class GCNLayer:
    def __init__(self, in_dim, out_dim):
        torch, nn = _import_torch()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, X, A):
        torch, _ = _import_torch()
        # X: (batch, nodes, features)
        # A: (nodes, nodes) — không tự loop trong buffer; cộng I ở đây
        n = A.size(0)
        I = torch.eye(n, device=A.device, dtype=A.dtype)
        A_tilde = A + I
        deg = A_tilde.sum(dim=1).clamp(min=1e-12)
        d_inv_sqrt = torch.pow(deg, -0.5)
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        out = self.W(A_norm @ X)
        return torch.relu(out)


# =========================
# GCN + LSTM (LSTM trên vector trạng thái cả đồ thị mỗi bước thời gian)
# =========================
class GCN_LSTM:
    """
    Mỗi bước thời gian: GCN trộn láng giềng không gian.
    Chuỗi thời gian: LSTM nhận (nodes * gcn_hidden) — toàn bộ nút sau GCN,
    giữ phụ thuộc không gian trong chuỗi thời gian (không tách LSTM theo từng trạm).
    """

    def __init__(
        self, A, num_nodes, in_features, hidden_dim=64, lstm_hidden=128, lstm_num_layers=2, dropout_p=0.3,
    ):
        torch, nn = _import_torch()
        self.num_nodes = num_nodes
        self.gcn_hidden = hidden_dim

        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        self.gcn = GCNLayer(in_features, hidden_dim)

        # Dropout sau GCN để giảm overfit vào nhiễu.
        self.dropout_gcn = nn.Dropout(dropout_p)

        # LSTM dùng num_layers=2 để dropout bên trong LSTM có hiệu lực.
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_num_layers,
            dropout=dropout_p if lstm_num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        # Dropout trên biểu diễn trước FC.
        self.dropout_lstm = nn.Dropout(dropout_p)
        self.fc = nn.Linear(lstm_hidden, 1)

    def register_buffer(self, name, tensor):
        torch, _ = _import_torch()
        setattr(self, name, tensor)

    def forward(self, x, target_idx=None):
        """
        x: (batch, time, nodes, features)
        target_idx:
          - None: trả về dự đoán cho tất cả nodes => (batch, nodes)
          - int: chỉ trả về dự đoán 1 node mục tiêu => (batch,)
        """
        torch, _ = _import_torch()
        batch, time_steps, nodes, _ = x.shape

        # 1) GCN cho từng bước thời gian: (batch, time, nodes, hidden)
        gcn_out = []
        for t in range(time_steps):
            xt = x[:, t, :, :]  # (batch, nodes, features)
            ht = self.gcn.forward(xt, self.A)  # (batch, nodes, hidden)
            ht = self.dropout_gcn(ht)
            gcn_out.append(ht)
        gcn_seq = torch.stack(gcn_out, dim=1)  # (batch, time, nodes, hidden)

        # 2) LSTM theo từng node (mỗi node có chuỗi thời gian, nhưng đầu vào đã được GCN trộn k-lân-cận)
        #    (batch, nodes, time, hidden) -> (batch*nodes, time, hidden)
        gcn_seq = gcn_seq.permute(0, 2, 1, 3).contiguous()
        gcn_flat = gcn_seq.view(batch * nodes, time_steps, self.gcn_hidden)

        lstm_out, _ = self.lstm(gcn_flat)  # (batch*nodes, time, lstm_hidden)
        last_step = lstm_out[:, -1, :]  # (batch*nodes, lstm_hidden)
        last_step = self.dropout_lstm(last_step)

        # 3) Dự đoán cho từng node rồi chọn 1 node nếu cần
        pred_all = self.fc(last_step).view(batch, nodes)  # (batch, nodes)

        if target_idx is None:
            return pred_all
        return pred_all[:, target_idx]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        """Sets internal modules to evaluation mode."""
        self.gcn.W.eval()
        self.dropout_gcn.eval()
        self.lstm.eval()
        self.dropout_lstm.eval()
        self.fc.eval()

    def load_state_dict(self, state_dict):
        torch, _ = _import_torch()
        # Load state dict
        for name, param in state_dict.items():
            if hasattr(self, name):
                setattr(self, name, param)

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

def update_graph_costs(edges, costs, nodes, model, scaler, X_test, model_type='keras', A=None, node_to_idx=None, speed_limit_kmh=60.0, intersection_delay_s=30.0):
    dynamic_costs = {}
    edge_list = list(costs.keys())
    batch_size = len(edge_list)
    
    if model_type == 'gcn_lstm':
        # Import torch locally
        torch, _ = _import_torch()
        # GCN-LSTM predicts for all nodes simultaneously
        # Randomly sample sequences for prediction 
        random_indices = [random.randint(0, len(X_test) - 1) for _ in range(batch_size)]
        batch_sequences = np.array([X_test[i] for i in random_indices])
        
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
        new_time_seconds = calculate_edge_cost(predicted_flow, dist_km, speed_limit_kmh, intersection_delay_s)
        dynamic_costs[(node_a, node_b)] = new_time_seconds
        
    return dynamic_costs

def get_path_distance_km(path, static_costs):
    total_degrees = 0
    for i in range(len(path) - 1): # - 1 to avoid index out of bounds
        total_degrees += static_costs.get((path[i], path[i+1]), 0) # + 1 to get the actual distance
    return total_degrees * 111.0

def run_tbrgs(filename, origin, destination, model_name, k_routes=5, speed_limit=60.0, intersection_delay=30.0):
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
        data_file = f'preprocessed_data_{model_name}.pkl'
    
    model_path = os.path.join(base_dir, 'models', model_dir, model_file)
    scaler_path = os.path.join(base_dir, 'models', model_dir, scaler_file)
    data_path = os.path.join(base_dir, 'data', 'preprocessed', data_file)
    

    # Parse the map file to extract graph structure and static costs
    nodes, edges, static_costs, _, _ = parse_file(filename)

    # Load trained model and preprocessing objects
    if model_type == 'gcn_lstm':
        # Load PyTorch model
        torch, _ = _import_torch()
        with open(data_path, "rb") as f:
            gcn_data = pickle.load(f)
        A = gcn_data["A"]
        node_to_idx = gcn_data["node_to_idx"]
        num_nodes = len(node_to_idx)
        in_features = len(gcn_data["feature_cols"])
        
        model = GCN_LSTM(A, num_nodes, in_features)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
    
    # Update edge costs based on predicted traffic flow
    dynamic_costs = update_graph_costs(edges, static_costs, nodes, model, scaler, X_test, model_type, A if model_type == 'gcn_lstm' else None, node_to_idx if model_type == 'gcn_lstm' else None, speed_limit, intersection_delay)
    
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

if __name__ == "__main__":
    # Test the TBRGS system by finding routes from node 3001 to 3685
    # Change model_name below to test different models: lstm, bidirectional_lstm, gru, bidirectional_gru, gcn_lstm
    routes = run_tbrgs(filename="Assignment2B/map_data/boroodara_tbrgs_map_coordinates.txt", origin="3001", destination="3685", model_name="lstm") # default test value
    
    # Display all recommended routes with estimated travel times and paths
    for r in routes:
        print(f"Route {r['route']} | Est. Time: {r['estimated_time_mins']} mins | Path: {' -> '.join(map(str, r['path']))}")
