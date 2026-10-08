import numpy as np
from numba import njit

from numba_tsp.types import DistMatrix, IntArray


@njit
def eulerian_circuit_from_double_tree(parent_tree: DistMatrix) -> IntArray:
    """
    Optimized Eulerian circuit of an MST assuming Node 0 is the root
    and parent forms a single connected tree.

    Parameters:
        parent (1D np.ndarray of int32/int64):
            Parent array of length N. parent[0] is ignored (Root = 0).

    Returns:
        circuit (1D np.ndarray of int32):
            Array of length 2*N - 1 representing the Eulerian circuit.
    """
    n = len(parent_tree)

    # Step 1: Build First-Child / Next-Sibling adjacency structure in 1 pass
    # Iterating backward ensures children are visited in ascending numerical order
    head = np.full(n, -1, dtype=np.int32)
    next_sibling = np.empty(n, dtype=np.int32)

    for i in range(n - 1, 0, -1):
        p = parent_tree[i]
        next_sibling[i] = head[p]
        head[p] = i

    curr_child = head.copy()

    # Step 2: Iterative DFS Euler Tour
    circuit = np.empty(2 * n - 1, dtype=np.int32)
    stack = np.empty(n, dtype=np.int32)

    top = 0
    stack[0] = 0
    circuit[0] = 0
    pos = 1

    while top >= 0:
        u = stack[top]
        v = curr_child[u]

        if v != -1:
            # Advance child pointer and move down to child v
            curr_child[u] = next_sibling[v]
            top += 1
            stack[top] = v
            circuit[pos] = v
            pos += 1
        else:
            # Backtrack to parent
            top -= 1
            if top >= 0:
                circuit[pos] = stack[top]
                pos += 1

    return circuit


@njit
def eulerian_shortcutting(eulerian_tour):
    """
    Shortcuts an Eulerian tour by removing duplicate node visits,
    yielding a TSP permutation (Hamiltonian path/cycle).

    Parameters:
        eulerian_tour (1D np.ndarray): Eulerian circuit array of length 2*N - 1.

    Returns:
        tour (1D np.ndarray): Permutation array of length N containing unique
                              vertices in order of first visit.
    """
    n = (len(eulerian_tour) + 1) // 2
    tour = np.empty(n, dtype=eulerian_tour.dtype)
    visited = np.zeros(n, dtype=np.bool_)

    idx = 0
    for v in eulerian_tour:
        if not visited[v]:
            visited[v] = True
            tour[idx] = v
            idx += 1

    return tour
