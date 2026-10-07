import numpy as np
from numba import njit

from .types import DistMatrix, IntArray


@njit(fastmath=True)
def prim_mst(dist_matrix: DistMatrix) -> IntArray:
    """
    Compute the Minimum Spanning Tree (MST) of a graph represented by a distance matrix using
    Prim's algorithm. The graph is assumed to be undirected and connected.
    """
    n = dist_matrix.shape[0]

    min_dist = np.full(n, np.inf, dtype=dist_matrix.dtype)
    parent = np.empty(n, dtype=np.int32)

    unvisited = np.arange(1, n, dtype=np.int32)
    num_unvisited = n - 1

    min_dist[0] = 0
    parent[0] = -1  # Root node has no parent

    # Step 0: Pre-relax neighbors of root node 0 without searching
    u = 0
    for slot in range(num_unvisited):
        v = unvisited[slot]
        w = dist_matrix[u, v]
        if w < min_dist[v]:
            min_dist[v] = w
            parent[v] = u

    # Main Prim loop processes remaining n-1 nodes
    while num_unvisited > 0:
        u_slot = -1
        min_val = np.inf

        # Single combined pass: relax distances AND find next minimum node
        for slot in range(num_unvisited):
            v = unvisited[slot]
            w = dist_matrix[u, v]

            if w < min_dist[v]:
                min_dist[v] = w
                parent[v] = u

            # Evaluate candidate minimum immediately after relaxation
            d = min_dist[v]
            if d < min_val:
                min_val = d
                u_slot = slot

        u = unvisited[u_slot]

        # Swap-and-pop removal in O(1)
        num_unvisited -= 1
        unvisited[u_slot] = unvisited[num_unvisited]

    return parent


@njit(fastmath=True)
def parent_to_edge_count(parent: IntArray) -> IntArray:
    """
    Given a parent array representing a tree, return an array where each index i contains the number
    of edges connected to node i. The root node (index 0) will have 0 edges. Assumes that the root
    node is at index 0 and that the parent array is valid (i.e., it represents a tree).
    """

    n = parent.shape[0]
    edge_count = np.ones(n, dtype=parent.dtype)
    edge_count[0] = 0  # Root node has no parent, so it has no edge

    for v in range(1, n):
        u = parent[v]
        edge_count[u] += 1

    return edge_count
