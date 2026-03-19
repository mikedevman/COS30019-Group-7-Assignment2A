import sys
from parser import parse_file
from dfs import dfs_search
from bfs import bfs 
from gbfs import gbfs
from astar import astar
from ucs import ucs
from weightedastar import weighted_astar

def abbrevation(method):
    if method == "DFS":
        return "Depth-First Search"
    elif method == "BFS":
        return "Breadth-First Search"
    elif method == "GBFS":
        return "Greedy Best-First Search"
    elif method == "AS":
        return "A*"
    elif method == "CUS1":
        return "Uniform Cost Search"
    elif method == "CUS2":
        return "Weighted A*"
    else:
        return method

def main():
    filename = sys.argv[1]
    method = sys.argv[2].upper()
    visualize = len(sys.argv) > 3 and sys.argv[3].lower() == "visualize"

    nodes, edges, costs, origin, destinations = parse_file(filename)

    if filename is None:
        print("No file provided")
        exit()

    if method == "DFS":
        goal, number_of_nodes, path, expanded = dfs_search(edges, origin, destinations)
    elif method == "BFS":
        number_of_nodes, path, expanded = bfs(origin, destinations, edges)
    elif method == "GBFS":  
        number_of_nodes, path, expanded = gbfs(origin, destinations, edges, nodes)
    elif method == "AS":
        number_of_nodes, path, expanded = astar(origin, destinations, edges, costs, nodes)
    elif method == "CUS1":
        number_of_nodes, path, expanded = ucs(origin, destinations, edges, costs, nodes)
    elif method == "CUS2":
        number_of_nodes, path, expanded = weighted_astar(origin, destinations, edges, costs, nodes)
    else:
        print("No method implemented")
        exit()

    if number_of_nodes and path is not None and len(path) > 0:
        print(f"Filename: {filename}, Method: {method} ({abbrevation(method)})")
        print(f"Goal: {path[-1]}, Number of nodes expanded: {number_of_nodes}") # [-1] to get the last element of the path 
        print(f"Path: {' -> '.join(map(str, path))}")
    else:
        print(filename, method)
        print("Search failed")

    if visualize:
        from visualize import launch
        import re, os
        base_filename = os.path.basename(filename)
        if base_filename == "PathFinder-test.txt":
            test_case_number = 0
        else:
            match = re.search(r"PathFinder-test-(\d+)\.txt", base_filename)
            test_case_number = int(match.group(1)) if match else filename
        launch(nodes, edges, origin, destinations, path or [], expanded, test_case_number=test_case_number, method=method)

if __name__ == "__main__":
    main()