from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def build_adjacency_matrix(df_long, k=3):
    """
    Build adjacency matrix based on spatial coordinates (latitude, longitude).
    Note: DF_long data needs to have 'lat', 'lon' columns.
    """
    nodes = sorted(df_long['SCATS_ID'].unique())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    coords = df_long.groupby('SCATS_ID')[['lat', 'lon']].mean().loc[nodes].values

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    N = len(nodes)
    A = np.zeros((N, N))

    for i in range(N):
        for j_idx in indices[i][1:]:
            dist = np.linalg.norm(coords[i] - coords[j_idx])
            A[i][j_idx] = 1 / (dist + 1e-6)
            A[j_idx][i] = A[i][j_idx]

    return A, node_to_idx

def build_tensor(df_long, node_to_idx, feature_cols):
    """
    Create Spatio-Temporal Tensor structure: (T_Time, N_Node, F_Feature)
    """
    times = sorted(df_long['Timestamp'].unique())
    nodes = list(node_to_idx.keys())

    T = len(times)
    N = len(nodes)
    F = len(feature_cols)

    tensor = np.zeros((T, N, F))
    
    # Optimize traversal time by group / pivot fast
    df_indexed = df_long.set_index(['Timestamp', 'SCATS_ID'])[feature_cols]
    
    for t_idx, t in enumerate(times):
        try:
            # Get data at time t for all nodes
            df_t = df_indexed.loc[t]
            for node in df_t.index:
                n_idx = node_to_idx[node]
                tensor[t_idx, n_idx, :] = df_t.loc[node].values
        except KeyError:
            continue
            
    return tensor

def create_st_sequences(tensor, seq_len=96):
    """
    Create sliding window sequences advancing by TIME FRAME on ALL stations.
    Input: tensor (T, N, F)
    Output: 
       X shape (samples, time_steps, stations, features)
       y shape (samples, stations) - traffic volume of stations (assumed at column 0) at next step
    """
    X, y = [], []
    T = tensor.shape[0]
    
    for i in range(T - seq_len):
        X.append(tensor[i : i + seq_len])
        # Predict traffic (feature 0) for ALL STATIONS
        y.append(tensor[i + seq_len, :, 0])  
        
    return np.array(X), np.array(y)

def visualize_spatial_graph(A, node_to_idx, df_long):
    print("\n   => Opening spatial graph visualization map...")
    G = nx.Graph()
    nodes = list(node_to_idx.keys())

    coords = df_long.groupby("SCATS_ID")[["lat", "lon"]].first().loc[nodes]

    pos = {}
    for i, node in enumerate(nodes):
        G.add_node(i, label=str(node))
        pos[i] = (coords.iloc[i]["lon"], coords.iloc[i]["lat"])

    N = len(nodes)
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j])

    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=300,
        node_color="lightcoral",
        font_size=8,
        font_weight="bold",
        edge_color="gray",
        alpha=0.9,
    )
    
    plt.title("Spatial Graph of SCATS Intersections (KNN Adjacency Matrix)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("on")
    plt.show()