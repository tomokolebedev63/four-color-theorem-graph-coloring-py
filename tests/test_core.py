"""Tests for the four-coloring implementation."""

import unittest

from four_color_theorem_graph_coloring_py import (
    FourColorer,
    ColorResult,
    ColoringError,
)


class TestColoring(unittest.TestCase):
    def setUp(self):
        self.colorer = FourColorer()

    def _is_valid(self, graph, result):
        """Assert that the coloring uses at most four colors and
        respects all edge constraints."""
        self.assertLessEqual(result.num_colors, 4)
        self.assertEqual(len(result.colors), len(graph))
        for v, neighbors in graph.items():
            self.assertIn(v, result.colors)
            for u in neighbors:
                self.assertNotEqual(
                    result.colors[v],
                    result.colors[u],
                    f"adjacent {v!r} and {u!r} share a color",
                )

    def test_empty_graph(self):
        result = self.colorer.color({})
        self.assertEqual(result.num_colors, 0)
        self.assertEqual(result.colors, {})

    def test_single_vertex(self):
        g = {"a": []}
        result = self.colorer.color(g)
        self._is_valid(g, result)
        self.assertEqual(result.num_colors, 1)

    def test_single_edge(self):
        g = {"a": ["b"], "b": ["a"]}
        result = self.colorer.color(g)
        self._is_valid(g, result)
        self.assertEqual(result.num_colors, 2)

    def test_triangle(self):
        g = {"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]}
        result = self.colorer.color(g)
        self._is_valid(g, result)
        self.assertEqual(result.num_colors, 3)

    def test_k4(self):
        # K4 is the complete graph on four vertices, a planar graph.
        g = {
            "a": ["b", "c", "d"],
            "b": ["a", "c", "d"],
            "c": ["a", "b", "d"],
            "d": ["a", "b", "c"],
        }
        result = self.colorer.color(g)
        self._is_valid(g, result)
        self.assertEqual(result.num_colors, 4)

    def test_square_cycle(self):
        g = {
            "a": ["b", "d"],
            "b": ["a", "c"],
            "c": ["b", "d"],
            "d": ["a", "c"],
        }
        result = self.colorer.color(g)
        self._is_valid(g, result)
        # C4 is bipartite, so two colors suffice.
        self.assertEqual(result.num_colors, 2)

    def test_star(self):
        g = {"center": ["l1", "l2", "l3", "l4"],
             "l1": ["center"], "l2": ["center"],
             "l3": ["center"], "l4": ["center"]}
        result = self.colorer.color(g)
        self._is_valid(g, result)
        self.assertEqual(result.num_colors, 2)

    def test_k5_fails(self):
        # K5 is non-planar and provably requires 5 colors.
        g = {str(i): [str(j) for j in range(5) if j != i]
             for i in range(5)}
        with self.assertRaises(ColoringError):
            self.colorer.color(g)

    def test_bipartite_large(self):
        # A complete bipartite graph K_{3,3} is non-planar but is
        # 2-colorable; greedy should still succeed here because there
        # is no edge within either partition.
        left = [f"l{i}" for i in range(3)]
        right = [f"r{i}" for i in range(3)]
        g = {}
        for v in left:
            g[v] = right[:]
        for v in right:
            g[v] = left[:]
        result = self.colorer.color(g)
        self._is_valid(g, result)
        self.assertEqual(result.num_colors, 2)

    def test_self_loop_raises(self):
        g = {"a": ["a", "b"], "b": ["a"]}
        with self.assertRaises(ValueError):
            self.colorer.color(g)

    def test_asymmetric_edge_raises(self):
        g = {"a": ["b"], "b": []}
        with self.assertRaises(ValueError):
            self.colorer.color(g)

    def test_unknown_neighbor_raises(self):
        g = {"a": ["x"], "b": ["a"]}
        # 'x' is not a key, so we raise.
        with self.assertRaises(ValueError):
            self.colorer.color(g)

    def test_integer_keys(self):
        g = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
        result = self.colorer.color(g)
        self._is_valid(g, result)

    def test_disconnected_graph(self):
        g = {
            "a": ["b"], "b": ["a"],
            "c": ["d"], "d": ["c"],
        }
        result = self.colorer.color(g)
        self._is_valid(g, result)
        self.assertEqual(result.num_colors, 2)

    def test_wheel_with_six_spokes(self):
        # Wheel W_6: a 6-cycle plus a hub connected to all cycle vertices.
        # This is planar and 4-colorable. The greedy heuristic should
        # find a valid coloring.
        cycle = [f"c{i}" for i in range(6)]
        g = {"hub": list(cycle)}
        for i, v in enumerate(cycle):
            g[v] = [cycle[(i - 1) % 6], cycle[(i + 1) % 6], "hub"]
        result = self.colorer.color(g)
        self._is_valid(g, result)
        self.assertLessEqual(result.num_colors, 4)

    def test_returns_color_result(self):
        g = {"a": ["b"], "b": ["a"]}
        result = self.colorer.color(g)
        self.assertIsInstance(result, ColorResult)

    def test_color_values_in_range(self):
        g = {
            "a": ["b", "c", "d"],
            "b": ["a", "c", "d"],
            "c": ["a", "b", "d"],
            "d": ["a", "b", "c"],
        }
        result = self.colorer.color(g)
        for c in result.colors.values():
            self.assertIn(c, {0, 1, 2, 3})

    def test_petersen_fails(self):
        # The Petersen graph is non-planar and requires 3 colors,
        # but greedy largest-first with 4 colors should succeed.
        # Actually the Petersen graph is 3-colorable, so greedy
        # with 4 colors will succeed. Test that it does.
        g = {
            0: [1, 4, 5],
            1: [0, 2, 6],
            2: [1, 3, 7],
            3: [2, 4, 8],
            4: [0, 3, 9],
            5: [0, 7, 8],
            6: [1, 8, 9],
            7: [2, 5, 9],
            8: [3, 5, 6],
            9: [4, 6, 7],
        }
        result = self.colorer.color(g)
        self._is_valid(g, result)
        self.assertLessEqual(result.num_colors, 4)


if __name__ == "__main__":
    unittest.main()
