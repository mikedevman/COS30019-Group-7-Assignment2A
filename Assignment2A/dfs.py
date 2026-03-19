import sys

def dfs_search(graph, origin, destinations):
    # Stack stores tuples of: (current_node, path_so_far)
    frontier = [(origin, [origin])]
    explored = set()
    expanded = [] # list to keep track of the order of expanded nodes
    
    # Initialize node counter (starting with 1 for the origin node)
    nodes_created = 1 

    while frontier:
        # LIFO: Pop the last element added to the stack
        current_node, path = frontier.pop()

        # 1. Goal Check
        if current_node in destinations:
            expanded.append(current_node) # add the current node to the list of expanded nodes
            return current_node, nodes_created, path, expanded

        # 2. Mark as explored
        if current_node not in explored:
            explored.add(current_node)
            expanded.append(current_node) 

            # 3. Get neighbors (avoiding nodes we've already fully explored)
            neighbors = graph.get(current_node, [])
            unvisited_neighbors = [n for n in neighbors if n not in explored]

            # 4. Tie-breaking rule 
            # To expand in ascending order, the smallest node needs to be on top of the stack
            # Therefore, neighbors are sorted in descending order before pushing
            unvisited_neighbors.sort(reverse=True)

            # 5. Push to frontier
            for neighbor in unvisited_neighbors:
                frontier.append((neighbor, path + [neighbor]))
                nodes_created += 1

    return None, nodes_created, [], expanded
