def parse_file(file_path):
    Nodes = {}
    Edges = {}
    Costs = {}
    Origin = None
    Destinations = []

    mode = None

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip() # remove trailing whitespace
            if not line:
                continue

            # determine data type
            if line == "Nodes:":
                mode = "nodes" 
                continue
            elif line == "Edges:":
                mode = "edges"
                continue
            elif line == "Origin:":
                mode = "origin"
                continue
            elif line == "Destinations:":
                mode = "destinations"
                continue

            # parse
            if mode == "nodes":
                node, coordinate = line.split(":")
                node = int(node) 
                coordinate = coordinate.strip("() ")
                x, y = map(int, coordinate.split(","))
                Nodes[node] = (x, y)
                Edges[node] = [] # initialize the list of edges for the node
            elif mode == "edges":
                edge, cost = line.split(":")
                edge = edge.strip("()")
                start, end = map(int, edge.split(","))
                Edges[start].append(end) # add the end node to the to the start node's list of edges
                Costs[(start, end)] = int(cost) # store the cost for the edge
            elif mode == "origin":
                Origin = int(line)
            elif mode == "destinations":
                Destinations = list(map(int, line.split(";")))

    return Nodes, Edges, Costs, Origin, Destinations