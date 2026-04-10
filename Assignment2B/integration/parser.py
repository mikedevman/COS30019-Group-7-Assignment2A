def parse_file(file_path):
    # Parse a map file into node coordinates, adjacency lists, and edge costs
    Nodes = {}
    Edges = {}
    Costs = {}
    
    # Track whether we are reading the [Nodes] or [Edges] section
    mode = None

    with open(file_path, 'r') as file:
        # Read the file line-by-line and switch behaviour based on section headers
        for line in file:
            line = line.strip()
            # Skip empty/formatting lines and separators
            if not line or not any(c.isalnum() for c in line):
                continue
            if "Nodes" in line:
                # Start parsing node definitions
                mode = "nodes"
                continue
            elif "Edges" in line:
                # Start parsing edge definitions
                mode = "edges"
                continue

            parts = line.split()
            
            # Skip section metadata line (e.g., "<num_nodes> <num_edges>")
            if len(parts) < 3:
                continue

            if mode == "nodes":
                # Node line format: <id> <longitude> <latitude>
                node_id = parts[0] 
                longtitude = float(parts[1])
                latitude = float(parts[2])
                # Store coordinates as (lat, lng) to match Leaflet-style ordering
                Nodes[node_id] = (latitude, longtitude)
                # Initialise adjacency list for this node
                Edges[node_id] = [] 
                
            elif mode == "edges":
                # Edge line format: <start_id> <end_id> <cost>
                start = parts[0]
                end = parts[1]
                cost = float(parts[2])
                
                # Ensure start node exists in adjacency dict then add directed edge
                if start not in Edges: Edges[start] = []
                Edges[start].append(end)
                # Store edge travel cost keyed by (start, end)
                Costs[(start, end)] = cost

    # Return signature includes unused placeholders for compatibility with older callers
    return Nodes, Edges, Costs, None, []