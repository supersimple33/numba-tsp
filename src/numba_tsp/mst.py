import numpy as np
from numba import njit

from .types import DistMatrix, IntArray


@njit(fastmath=True)
def prim_mst(dist_matrix: DistMatrix) -> IntArray:
    n = dist_matrix.shape[0]

    min_dist = np.full(n, np.inf, dtype=dist_matrix.dtype)
    parent = np.empty(n, dtype=np.int64)

    unvisited = np.arange(1, n, dtype=np.int64)
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

        for slot in range(num_unvisited):
            node = unvisited[slot]
            if min_dist[node] < min_val:
                min_val = min_dist[node]
                u_slot = slot

        u = unvisited[u_slot]

        # Swap-and-pop removal in O(1)
        num_unvisited -= 1
        unvisited[u_slot] = unvisited[num_unvisited]

        # Relax remaining unvisited nodes
        for slot in range(num_unvisited):
            v = unvisited[slot]
            w = dist_matrix[u, v]
            if w < min_dist[v]:
                min_dist[v] = w
                parent[v] = u

    return parent
