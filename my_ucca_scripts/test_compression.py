from get_edges_from_mrp import test
from get_edges_from_mrp import get_mrp_edges
from compress_c import compress_c_edge


edges = get_mrp_edges(test)
compressed = compress_c_edge(edges)
