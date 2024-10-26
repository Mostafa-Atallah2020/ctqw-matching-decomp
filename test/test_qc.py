from src.graphs import MultiEdgeGraph, DiagonalEdgeGraph


def test_000_111_cxs():
    G = DiagonalEdgeGraph({("000", "111")})
    connections = G.connections

    assert connections == [(0, 1), (0, 2)]
