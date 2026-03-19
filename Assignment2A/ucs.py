import heapq

def ucs(origin, destinations, graph, costs, nodes):
    destination_set = set(destinations)

    counter = 0 # monotonically increasing insertion counter

    # heap entry: (g, node, counter, node)
    heap = [(0, origin, counter, origin)]
    g = {origin: 0} # best known path cost to each node
    visited = set() # set of expanded nodes
    parent = {origin: None} # dictionary to store parent of each node for path reconstruction
    expanded = [] # list to keep track of the order of expanded nodes

    number_of_nodes = 0 

    while heap:
        g_cur, current, _, _ = heapq.heappop(heap) # pop the node with the lowest g-cost from the heap

        if current in visited:
            continue
        visited.add(current)
        number_of_nodes += 1  # count each node when it is expanded

        if current in destination_set:
            # reconstruct path from origin to current
            path = [] # list to store the path from origin to destination
            node = current # start from the destination node
            while node is not None:
                path.append(node) # add the current node to the path
                node = parent[node] # move to the parent of the current node
            path.reverse() # reverse the path to get the correct order from origin to destination
            return number_of_nodes, path, expanded

        # Expand neighbours in ascending node-id order so that for equal g the smaller id is pushed first (and gets a lower counter too).
        for neighbour in sorted(graph[current]):
            if neighbour in visited:
                continue

            edge_cost = costs.get((current, neighbour), 1) # get the cost of the edge from current to neighbour, default to 1 if not specified
            tentative_g = g_cur + edge_cost # calculate the tentative g-cost to the neighbour through the current node

            if neighbour not in g or tentative_g < g[neighbour]: # if the neighbour has not been visited or a cheaper path to the neighbour is found
                g[neighbour] = tentative_g # update the best known g-cost to the neighbour
                parent[neighbour] = current # set the parent of the neighbour to the current node for path reconstruction
                counter += 1 # increment the counter for tie-breaking in the heap
                heapq.heappush(heap, (tentative_g, neighbour, counter, neighbour)) # push the neighbour onto the heap with its g-cost, node id, and counter for tie-breaking

    return number_of_nodes, None, expanded # return none if no path is found to any of the destinations
