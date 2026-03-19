import heapq
import math

def euclidean(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) # calculate the Euclidean distance between two coordinates

def heuristic(node, destinations, nodes):
    return min(euclidean(nodes[node], nodes[d]) for d in destinations) # calculate the heuristic value for a node as the minimum Euclidean distance to any of the destination nodes

def astar(origin, destinations, graph, costs, nodes):
    destination_set = set(destinations)
    h_start = heuristic(origin, destinations, nodes) # calculate the heuristic value for the origin node
    expanded = [] # list to keep track of the order of expanded nodes

    counter = 0  # monotonically increasing insertion counter

    # heap entry: (f, node, counter, node)
    heap = [(h_start, origin, counter, origin)]

    g = {origin: 0} # best known g-cost to each node
    
    visited = set() # set of expanded nodes
    parent = {origin: None} # dictionary to store parent of each node for path reconstruction

    number_of_nodes = 0 # count of expanded nodes

    while heap:
        f, current, _, _ = heapq.heappop(heap) # pop the node with the lowest f-cost from the heap

        if current in visited:
            continue
        visited.add(current)
        number_of_nodes += 1  # count each node when it is expanded
        expanded.append(current) # add the current node to the list of expanded nodes

        if current in destination_set:
            # reconstruct path
            path = [] # list to store the path from origin to destination
            node = current # start from the destination node
            while node is not None:
                path.append(node) # add the current node to the path
                node = parent[node] # move to the parent of the current node
            path.reverse() # reverse the path to get the correct order from origin to destination
            return number_of_nodes, path, expanded

        # expand neighbours in ascending node-id order so that for equal f the smaller id is pushed first (gets a lower counter too)
        for neighbour in sorted(graph[current]):
            if neighbour in visited:
                continue

            edge_cost = costs.get((current, neighbour), 1) # get the cost of the edge from current to neighbour, default to 1 if not specified
            tentative_g = g[current] + edge_cost # calculate the tentative g-cost to the neighbour through the current node

            if neighbour not in g or tentative_g < g[neighbour]: # if the neighbour has not been visited or a cheaper path to the neighbour is found
                g[neighbour] = tentative_g # update the best known g-cost to the neighbour
                parent[neighbour] = current # set the parent of the neighbour to the current node for path reconstruction
                h_n = heuristic(neighbour, destinations, nodes) # calculate the heuristic value for the neighbour
                f_n = tentative_g + h_n # calculate the f-cost for the neighbour
                counter += 1 # increment the counter for tie-breaking in the heap
                heapq.heappush(heap, (f_n, neighbour, counter, neighbour)) # push the neighbour onto the heap with its f-cost, node id, and counter for tie-breaking
    return number_of_nodes, None, expanded # return none if no path is found to any of the destinations
