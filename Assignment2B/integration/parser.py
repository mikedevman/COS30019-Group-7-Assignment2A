def parse_file(file_path):
    Nodes = {}
    Edges = {}
    Costs = {}
    
    mode = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or not any(c.isalnum() for c in line):
                continue
            if "Nodes" in line:
                mode = "nodes"
                continue
            elif "Edges" in line:
                mode = "edges"
                continue

            parts = line.split()
            
            # Skip the metadata line (e.g., "50 150")
            if len(parts) < 3:
                continue

            if mode == "nodes":
                node_id = parts[0] 
                longtitude = float(parts[1])
                latitude = float(parts[2])
                # Nodes[node_id] = (longtitude, latitude)
                Nodes[node_id] = (latitude, longtitude)
                Edges[node_id] = [] 
                
            elif mode == "edges":
                # Handle space-separated edges: start end cost
                start = parts[0]
                end = parts[1]
                cost = float(parts[2])
                
                if start not in Edges: Edges[start] = []
                Edges[start].append(end)
                Costs[(start, end)] = cost

    return Nodes, Edges, Costs, None, []