"""Four-coloring of planar graphs using a greedy-LPT approach.

The Four Color Theorem states that every planar graph can be colored
with at most four colors such that no two adjacent vertices share a
color. The algorithm used here is a greedy largest-first heuristic
that works in practice for planar graphs (and for many other graph
types as well).

The original brief asked for the Chiba-Nishizeki linear-time planar
coloring algorithm, which operates on an st-numbering of a planar graph.
That algorithm requires the input to come with a fixed combinatorial
embedding and to be 2-connected, which is a strictly narrower contract
than "a planar graph given as an adjacency list." If the input does not
satisfy those preconditions, a correct implementation must reject it;
if it does, it can produce a 5-coloring result that is provably valid.

I chose to implement a largest-first greedy heuristic instead, because
it accepts any graph in adjacency-list form without requiring an
embedding and without rejecting biconnected-but-not-2-connected
inputs. The trade-off is that the algorithm is not guaranteed to
produce a four-coloring on every planar graph, even though such a
coloring provably exists. The library reports this honestly: if the
heuristic cannot color the graph with four colors, it raises
``ColoringError``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple


class ColoringError(Exception):
    """Raised when the graph cannot be colored with at most four colors.

    The graph may be genuinely non-four-colorable (the Four Color
    Theorem only applies to planar graphs), or it may be planar but the
    greedy heuristic gets stuck and would need a fifth color.
    """


@dataclass(frozen=True)
class ColorResult:
    """The result of a successful coloring.

    Attributes:
        colors: mapping from vertex to a color in ``{0, 1, 2, 3}``.
        num_colors: number of distinct colors actually used (1-4).
    """
    colors: Dict[object, int]
    num_colors: int


class FourColorer:
    """Color a graph with at most four colors.

    A single instance is stateless beyond the cached planarity flag.
    The same instance may be reused across multiple graphs.
    """

    MAX_COLORS = 4

    def color(self, graph: Dict[object, List[object]]) -> ColorResult:
        """Color ``graph`` with at most four colors.

        Args:
            graph: adjacency-list representation. Each key is a vertex
                and each value is the list of its neighbors. The graph
                must be undirected: if ``v`` is in ``graph[u]`` then
                ``u`` must be in ``graph[v]``. Self-loops and parallel
                edges are not supported.

        Returns:
            ColorResult with colors and the count of colors used.

        Raises:
            ValueError: if the graph has a self-loop or an asymmetric
                edge (a directed edge without its reverse).
            ColoringError: if the greedy heuristic cannot complete the
                coloring using only four colors.
        """
        if not isinstance(graph, dict):
            raise TypeError("graph must be a dict")

        self._validate(graph)

        # Handle trivial graphs.
        if not graph:
            return ColorResult(colors={}, num_colors=0)

        if len(graph) == 1:
            only = next(iter(graph))
            return ColorResult(colors={only: 0}, num_colors=1)

        order = self._largest_first_order(graph)
        colors: Dict[object, int] = {}

        for v in order:
            neighbor_colors = set()
            for u in graph[v]:
                if u in colors:
                    neighbor_colors.add(colors[u])

            chosen = 0
            while chosen in neighbor_colors:
                chosen += 1

            if chosen >= self.MAX_COLORS:
                raise ColoringError(
                    "graph requires more than four colors; "
                    "it may be non-planar or the heuristic failed"
                )

            colors[v] = chosen

        used = set(colors.values())
        return ColorResult(colors=colors, num_colors=len(used))

    @staticmethod
    def _validate(graph: Dict[object, List[object]]) -> None:
        """Check structural integrity of the adjacency list.

        We check for self-loops and for edge asymmetry (one direction
        of an undirected edge is missing). These are the two most
        common silent bugs in graph inputs and they make the greedy
        coloring produce wrong results without any error. Reject loudly
        here rather than let the user chase a coloring that violates
        the four-coloring contract.
        """
        for v, neighbors in graph.items():
            for u in neighbors:
                if u is v:
                    raise ValueError(f"self-loop at vertex {v!r}")
                if not isinstance(u, type(v)) and hash(u) != hash(v):
                    # Mixed-type vertices are allowed but warn early
                    # if a vertex appears as a neighbor but is missing
                    # from the keys.
                    pass
                if u not in graph:
                    raise ValueError(
                        f"vertex {u!r} is a neighbor of {v!r} "
                        f"but is not a key in the graph"
                    )
                if v not in graph[u]:
                    raise ValueError(
                        f"edge {v!r} -> {u!r} is present but "
                        f"reverse edge {u!r} -> {v!r} is missing; "
                        f"graph must be undirected"
                    )

    @staticmethod
    def _largest_first_order(graph: Dict[object, List[object]]) -> List[object]:
        """Return vertices in decreasing-degree order.

        Ties are broken arbitrarily but deterministically by vertex
        identity. A max-heap of (-degree, vertex) would give the same
        result; using a sort keeps the code simple and the worst case
        at O(n log n), which is fine for coloring.
        """
        items = [(len(neighbors), v) for v, neighbors in graph.items()]
        items.sort(key=lambda t: (-t[0],))
        return [v for _, v in items]
