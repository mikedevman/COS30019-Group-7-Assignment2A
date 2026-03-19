# Test Cases

**Test Case 0:** Sample test file provided in the assignment. Origin: 2, Destination: 5, 4.

**Test Case 1:** A straight linear chain of 5 nodes. Origin: 1, Destination: 5.

**Test Case 2:** A single-node graph with no edges. Origin: 1, Destination: 1.

**Test Case 3:** A simple connected graph with 4 nodes in a line. Origin: 1, Destination: 4.

**Test Case 4:** The origin node is also the destination. Origin: 1, Destination: 1.

**Test Case 5:** Two destinations reachable from the origin on separate branches. Origin: 1, Destinations: 4, 5.

**Test Case 6:** A directed graph where the destination is reachable via an intermediate node. Origin: 1, Destination: 3.

**Test Case 7:** A graph where multiple paths exist to the destination but with different costs. Origin: 1, Destination: 4.

**Test Case 8:** The graph contains a cycle. Tests that algorithms don't loop infinitely. Origin: 1, Destination: 5.

**Test Case 9:** A large grid graph of 15 nodes used to compare node expansion counts across algorithms. Origin: 1, Destination: 15.

**Test Case 10:** Two equal-cost paths to the same destination. Tests tie-breaking by node number. Origin: 1, Destination: 4.



