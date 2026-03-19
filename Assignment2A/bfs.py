from collections import deque

def bfs(origin, destinations, graph):
    queue = deque([origin]) # FIFO queue
    visited = set([origin]) # set of visited nodes
    parent = {origin: None} # dictionary to store parent of each node
    expanded = [] # list to keep track of the order of expanded nodes

    number_of_nodes = 1

    while queue:
        current = queue.popleft() # remove the first node from the queue
        expanded.append(current) # add the current node to the list of expanded nodes
        if current in destinations:
            path = []
            while current is not None:
                path.append(current) # add the current node to the path
                current = parent[current] # move to the parent of the current node
            path.reverse() # reverse the path to get the correct order from origin to destination
            return number_of_nodes, path, expanded
        for neighbor in sorted(graph[current]): # for each neighbor of the current node, sorted alphabetically
            if neighbor not in visited:
                visited.add(neighbor) # mark the neighbor as visited
                parent[neighbor] = current # set the parent of the neighbor to the current node
                queue.append(neighbor)
                number_of_nodes += 1 # increment the number of nodes visited
    return number_of_nodes, None, expanded # return none if no path is found to any of the destinations
