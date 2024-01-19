from manim import *


class TestScene(Scene):
    def construct(self):
        graph = Graph(
            vertices=[1, 2, 3, 4, 5],
            edges=[
                (1, 2),
                (1, 3),
                (1, 4),
                (1, 5),
                (2, 3),
                (2, 4),
                (2, 5),
                (3, 4),
                (3, 5),
                (4, 5),
            ],
            vertex_type=Circle,
            vertex_config={"radius": 0.25},
        )
        # default behavior; edges are created one at a time
        self.play(Create(graph), run_time=4)
        self.clear()
        # manually created the vertices, works as expected
        self.play(
            AnimationGroup(
                *(Create(vertex) for vertex in graph.vertices.values()),
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        # manually create the edges, edges appear all at once?
        self.play(
            AnimationGroup(
                *(Create(edge) for edge in graph.edges.values()),
                lag_ratio=0.1,
            ),
            run_time=2,
        )
        self.pause(3)
