import heapq
import math

def euclidean(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) # calculate the Euclidean distance between two coordinates

def heuristic(node, destinations, nodes):
    return min(euclidean(nodes[node], nodes[d]) for d in destinations) # calculate the heuristic value for a node as the minimum Euclidean distance to any of the destination nodes

def gbfs(origin, destinations, graph, nodes):
    destination_set = set(destinations)
    h_start = heuristic(origin, destinations, nodes) # calculate the heuristic value for the origin node
    counter = 0  # monotonically increasing insertion counter
    # heap entry: (h, node, counter, node) — node appears twice so heapq never needs to compare the parent dictionary (non-comparable)
    heap = [(h_start, origin, counter, origin)]
    visited = set()
    parent = {origin: None}
    expanded = [] # list to keep track of the order of expanded nodes

    number_of_nodes = 0

    while heap: 
        h, current, _, _ = heapq.heappop(heap) # pop the node with the lowest h-cost from the heap
        expanded.append(current) # add the current node to the list of expanded nodes

        if current in visited:
            continue
        visited.add(current) # mark the current node as visited (expanded)
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

        # Expand neighbours in ascending node-id order so that when two neighbours share the same h value the smaller id is pushed first and therefore 
        # gets a smaller counter — satisfying both tie-breaking rules simultaneously.
        for neighbour in sorted(graph[current]):
            if neighbour not in visited:
                if neighbour not in parent:
                    parent[neighbour] = current # set the parent of the neighbour to the current node for path reconstruction
                counter += 1 # increment the counter for tie-breaking in the heap
                h_n = heuristic(neighbour, destinations, nodes) # calculate the heuristic value for the neighbour
                heapq.heappush(heap, (h_n, neighbour, counter, neighbour)) # push the neighbour onto the heap with its h-cost, node id, and counter for tie-breaking

    return number_of_nodes, None, expanded # return none if no path is found to any of the destinations
