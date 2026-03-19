import parser
import math
import heapq

weight = 1.2

def weighted_astar(origin, destinations, graph, costs, nodes):
    start = origin
    goal = destinations[0] if len(destinations) == 1 else destinations[0]  # default to first destination

    open_list = [] # priority queue for nodes to explore
    heapq.heappush(open_list, (0, start)) # (f_cost, node)
    came_from = {} # to reconstruct path

    g_cost = {start: 0} # cost from start to current node
    nodes_expanded = 0
    expanded = [] # list to keep track of the order of expanded nodes

    destination_set = set(destinations)

    while open_list: 
        # retrieve node with lowest f_cost
        current_f, current = heapq.heappop(open_list)
        nodes_expanded += 1
        expanded.append(current) # add the current node to the list of expanded nodes
        # reconstruct path 
        if current in destination_set:
            path = [] # add nodes to path
            while current in came_from: 
                path.append(current)
                current = came_from[current]
            path.append(start) # add start node to path
            path.reverse() # reverse path to get correct order from start to goal
            return nodes_expanded, path, expanded
        # check neighbors of current node
        for neighbor in graph[current]:
            cost = costs.get((current, neighbor), 1)
            # calculate g_cost for neighbor
            new_g = g_cost[current] + cost
            # if neighbor is not in visited or found a cheaper path to neighbor
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                # calculate heuristic cost to goal 
                h = min(heuristic(neighbor, d, nodes) for d in destinations)
                # weighted astar formula: f = g + weight * h
                f = new_g + weight * h
                # add neighbor into priority queue
                heapq.heappush(open_list, (f, neighbor))
                # record parent for path reconstruction
                came_from[neighbor] = current 
    return nodes_expanded, None, expanded # return none if no path is found to any of the destinations

def heuristic(node, goal, nodes):
    x1, y1 = nodes[node]
    x2, y2 = nodes[goal]

    # euclidean distance
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)