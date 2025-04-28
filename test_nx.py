"""Test NetworkX imports and functionality."""
import sys
import os
import networkx as nx

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Try to import our path finding module
from src.algorithms.network_path_finding import build_graph_from_grid

print(f"NetworkX version: {nx.__version__}")
print("Module imported successfully!")

# Create a simple graph
G = nx.DiGraph()
G.add_node(1)
G.add_node(2)
G.add_edge(1, 2, weight=0.5)

print(f"Graph created with {len(G.nodes())} nodes and {len(G.edges())} edges")
print(f"Shortest path: {nx.shortest_path(G, 1, 2, weight='weight')}")

print("Test completed successfully!")