import numpy as np

from .eulerian_circuits import eulerian_circuit_from_double_tree, eulerian_shortcutting
from .mst import parent_to_edge_count, prim_mst
from .types import DistMatrix


def double_tree_approx(dist_matrix: DistMatrix):
    parent = prim_mst(dist_matrix)
    eulerian_circuit = eulerian_circuit_from_double_tree(parent)
    return eulerian_shortcutting(eulerian_circuit)

