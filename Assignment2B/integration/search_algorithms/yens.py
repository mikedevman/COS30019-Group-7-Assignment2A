import copy
from .astar import astar

def get_path_cost(path, costs):
    # Calculates the total travel time of a given path
    if not path or len(path) < 2:
        return 0
        
    total_cost = 0
    for i in range(len(path) - 1):
        # Default to a massive penalty if an edge is somehow missing
        total_cost += costs.get((path[i], path[i+1]), float('inf'))
    return total_cost

def yens_k_shortest_paths(origin, destinations, graph, costs, nodes, K=5):
    # Finds the Top-K fastest paths from origin to destination.
    A = [] # Final list of our top K paths
    
    # 1. Find the absolute fastest route using your A*
    _, first_path, _ = astar(origin, destinations, graph, costs, nodes)
    
    if not first_path:
        return [] 
        
    A.append(first_path)
    B = [] # Potential alternative paths
    
    # 2. Loop to find the remaining routes (up to K)
    for k in range(1, K):
        prev_path = A[k-1]
        
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            root_path = prev_path[:i+1]
            
            # Copy the graph so we can temporarily block roads
            temp_graph = copy.deepcopy(graph)
            
            # Block roads that we already used in previous paths 
            # to force A* to find a new way around
            for path in A:
                if len(path) > i and path[:i+1] == root_path:
                    v = path[i+1]
                    if spur_node in temp_graph and v in temp_graph[spur_node]:
                        temp_graph[spur_node].remove(v)
            
            # Prevent loops by removing root path nodes
            for root_path_node in root_path[:-1]:
                if root_path_node in temp_graph:
                    temp_graph[root_path_node] = []
            
            # 3. Run A* on the temporarily restricted map
            _, spur_path, _ = astar(spur_node, destinations, temp_graph, costs, nodes)
            
            if spur_path:
                total_path = root_path[:-1] + spur_path
                path_cost = get_path_cost(total_path, costs)
                
                if total_path not in [p for cost, p in B]:
                    B.append((path_cost, total_path))
                    
        if not B:
            break # No more physical routes exist
            
        # 4. Pick the fastest alternative and add it to our final list
        B.sort(key=lambda x: x[0]) 
        best_candidate = B.pop(0)[1]
        A.append(best_candidate)
        
    return A