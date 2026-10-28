# four-color-theorem-graph-coloring-py

Colors an undirected graph with at most four colors using a greedy largest-first heuristic. Returns the coloring as a mapping from vertex to color, along with the number of colors actually used.

## Usage

```python
from four_color_theorem_graph_coloring_py import FourColorer, ColorResult, ColoringError

colorer = FourColorer()

graph = {
    "a": ["b", "c", "d"],
    "b": ["a", "c", "d"],
    "c": ["a", "b", "d"],
    "d": ["a", "b", "c"],
}

try:
    result: ColorResult = colorer.color(graph)
    print(result.colors)     # e.g. {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    print(result.num_colors)  # 4
except ColoringError as e:
    print(f"cannot four-color this graph: {e}")
```

The input is an adjacency-list `dict` mapping each vertex to its list of neighbors. The graph must be undirected and may not contain self-loops.

## Why this exists

The Four Color Theorem guarantees that every planar graph admits a coloring with at most four colors. The linear-time algorithm of Chiba and Nishizeki computes such a coloring but requires the input graph to come with a fixed combinatorial embedding and to be 2-connected — a strictly narrower contract than "a planar graph given as an adjacency list."

This library implements a greedy largest-first heuristic instead. It accepts any graph in adjacency-list form without requiring an embedding and without rejecting biconnected-but-not-2-connected inputs. The trade-off is that the algorithm is not guaranteed to produce a four-coloring on every planar graph, even though such a coloring provably exists. When the heuristic cannot complete the coloring using only four colors, the library raises `ColoringError` rather than silently producing an invalid coloring.

## The awkward edge

A planar graph that the heuristic fails to four-color will raise `ColoringError` even though a valid four-coloring exists. If you need a guaranteed four-coloring on every planar input, you must either provide the combinatorial embedding and use a dedicated linear-time planar-coloring implementation, or fall back to a slower exact method.

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```

## Exported names

- `FourColorer` — the coloring class; call `.color(graph)` on an instance.
- `ColorResult` — frozen dataclass with `colors` (dict from vertex to int in `{0,1,2,3}`) and `num_colors` (int).
- `ColoringError` — raised when the graph cannot be colored with at most four colors.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

