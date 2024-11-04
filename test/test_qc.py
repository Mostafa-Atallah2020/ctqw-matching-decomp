from src.graphs import MultiEdgeGraph


def test_010_101_cxs():
    edges = {("010", "101")}
    G = MultiEdgeGraph(edges)
    assert G.connections == [(0, 1), (0, 2)]


def test_000_111_cxs():
    edges = {("000", "111")}
    G = MultiEdgeGraph(edges)
    assert G.connections == [(0, 1), (0, 2)]


def test_001_110_cxs():
    edges = {("001", "110")}
    G = MultiEdgeGraph(edges)
    assert G.connections == [(0, 1), (0, 2)]


def test_001_100_cxs():
    edges = {("001", "100")}
    G = MultiEdgeGraph(edges)
    assert G.connections == [(0, 2)]


def test_010_100_cxs():
    edges = {("010", "100")}
    G = MultiEdgeGraph(edges)
    assert G.connections == [(0, 1)]


def test_000_011_cxs():
    edges = {("000", "011")}
    G = MultiEdgeGraph(edges)
    assert G.connections == [(1, 2)]


def test_011_110_cxs():
    edges = {("011", "110")}
    G = MultiEdgeGraph(edges)
    assert G.connections == [(0, 2)]


def test_100_111_cxs():
    edges = {("100", "111")}
    G = MultiEdgeGraph(edges)
    assert G.connections == [(1, 2)]


def test_010_111_cxs():
    edges = {("010", "111")}
    G = MultiEdgeGraph(edges)
    assert G.connections == [(0, 2)]


def test_000_101_cxs():
    edges = {("000", "101")}
    G = MultiEdgeGraph(edges)
    assert G.connections == [(0, 2)]


def test_000_110_cxs():
    edges = {("000", "110")}
    G = MultiEdgeGraph(edges)
    assert G.connections == [(0, 1)]
