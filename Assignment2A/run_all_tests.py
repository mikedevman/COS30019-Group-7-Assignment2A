import sys
import os
from search import main as search_main
from termcolor import colored

def run_all_tests(method):
    test_dir = "tests"
    test_files = [f for f in os.listdir(test_dir) if f.startswith("PathFinder-test") and f.endswith(".txt")]

    test_files.sort()
    if test_files:
        last_file = test_files.pop()
        test_files.insert(0, last_file) # move the last file to the front of the list
        def correct_order_test10(f):
            digits = ''.join(filter(str.isdigit, f)) # extract digits from the filename
            return int(digits) if digits else -1 # -1 to ensure files without digits are sorted before those with digits
        test_files.sort(key=correct_order_test10) # correct the order of test 10
    passed = 0
    total = 11
    for test_file in test_files:
        # extract test case number from filename
        import re
        match = re.search(r"PathFinder-test-(\d+)\.txt", test_file)
        if match:
            test_case_number = match.group(1)
            print(colored(f"\nRunning Test Case {test_case_number} with method {method.upper()}:", "yellow"))
        else:
            print(colored(f"\nRunning Default Test Case with method {method.upper()}:", "yellow"))
        sys.argv = ["search.py", os.path.join(test_dir, test_file), method.upper()]
        try:
            search_main()
            passed += 1
        except Exception as e:
            print(f"Error: {e}")
    print(colored(f"\nTest Cases Passed: {passed}/{total}", "green") if passed == total else colored(f"\nTest Cases Passed: {passed}/{total}", "red"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(colored("\nSearch methods:", "yellow"))
        print("DFS (Depth-First Search)")
        print("BFS (Breadth-First Search)")
        print("GBFS (Greedy Best-First Search)")
        print("AS (A*)")
        print("CUS1 (Uniform Cost Search)")
        print("CUS2 (Weighted A*)")
        method = input("\nEnter search method: ")
    else:
        method = sys.argv[1]
    run_all_tests(method)
