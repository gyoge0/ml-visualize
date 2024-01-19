from collections.abc import Iterable
from dataclasses import dataclass
from typing import Tuple

from manim import *

import tensorflow as tf
from tensorflow import keras
import numpy as np


@dataclass(eq=True, frozen=True)
class Node:
    layer_idx: int
    neuron_idx: int


class Neuron(Circle):
    def __init__(self, *args, **kwargs):
        super().__init__(
            radius=0.125,
            *args,
            **kwargs,
        )
        pass


def _get_nodes_from_model(model: keras.models.Model):
    nodes: Dict[Tuple[int, int], Node] = {}

    for l_idx, layer in enumerate(model.layers):
        match layer:
            case keras.layers.Dense():
                count = layer.units
            case _:
                count = layer.output_shape[1]

        for n_idx in range(count):
            nodes[(l_idx, n_idx)] = Node(l_idx, n_idx)

    return nodes


def _get_layout_from_nodes(
    nodes: Dict[Tuple[int, int], Node],
    layer_scale: Tuple[float] | float = 2,
    network_scale: float = 2,
):
    max_l_idx = 0
    max_n_idx = {}

    for l_idx, n_idx in nodes.keys():
        if l_idx > max_l_idx:
            max_l_idx = l_idx
        if n_idx > max_n_idx.get(l_idx, -1):
            max_n_idx[l_idx] = n_idx

    if isinstance(layer_scale, Iterable):
        layer_scale = {idx: i for idx, i in enumerate(layer_scale)}
    else:
        layer_scale = {i: layer_scale for i in range(max_l_idx)}

    layout = {}

    for (layer, neuron), value in nodes.items():
        # get how tall this layer is allowed to be
        neuron_scale = layer_scale.get(layer, 1)
        neuron = (neuron / max_n_idx[layer]) * neuron_scale - (neuron_scale / 2)
        layer = (layer / max_l_idx) * network_scale - (network_scale / 2)
        layout[value] = [layer, neuron, 0]

    return layout


class NeuralNetwork(Graph):
    def __init__(
        self,
        model: keras.models.Model,
        layer_scale: Tuple[float] | float = 4,
        network_scale: float = 2,
        **kwargs,
    ):
        self.model = model

        self.nodes = _get_nodes_from_model(self.model)
        vertices = list(self.nodes.values())
        edges = self.edges_from_weights()

        # turns Node: (x, y) to [x, y, 0]: Node
        layout = _get_layout_from_nodes(
            self.nodes,
            layer_scale,
            network_scale,
        )

        super().__init__(
            vertices=vertices,
            edges=edges,
            edge_config={"stroke_width": DEFAULT_STROKE_WIDTH / 2},
            layout=layout,
            vertex_type=Neuron,
            **kwargs,
        )

    def edges_from_weights(self):
        new_edges = []

        model_layers = self.model.layers

        for layer_idx, layer in enumerate(model_layers):
            # we connect the previous to the current so we skip this
            if layer_idx == 0:
                continue

            raw_weights: np.array = layer.weights[0].numpy()
            for (from_idx, to_idx), weight in np.ndenumerate(raw_weights):
                from_node = self.nodes[(layer_idx - 1, from_idx)]
                to_node = self.nodes[(layer_idx, to_idx)]
                new_edges.append((from_node, to_node))

        return new_edges
        # self.remove_edges(*(edge for edge in self.edges if edge not in new_edges))
        # self.add_edges(*(edge for edge in new_edges if edge not in self.edges))
        # I don't know if I can clear the values and have them stay the same
        # keys_to_remove = [key for key in self.edges if key not in new_edges]
        # for key in keys_to_remove:
        #     del self.edges[key]
        # self.edges.update(
        #     {key: new_edges[key] for key in new_edges if key not in self.edges}
        # )

    @override_animation(Create)
    def _create_override(self, **kwargs):
        return Succession(
            AnimationGroup(
                *(Create(vertex) for vertex in self.vertices.values()),
                lag_ratio=kwargs.get("neuron_lag_ratio", 0.025),
            ),
            AnimationGroup(
                *(Create(edge) for edge in self.edges.values()),
                # for some reason, if a lag ratio of 0.01 is used, then this glitches out
                lag_ratio=kwargs.get("weight_lag_ratio", 0.01 + 1e-9),
            ),
        )
