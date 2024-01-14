from typing import Tuple

from manim import *
from typing_extensions import Self
import numpy as np


class Neuron(Circle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EllipsisBetween(DashedLine):
    def __init__(self, start: Neuron, end: Neuron, *args, **kwargs):
        self.start = start
        self.end = end
        super().__init__(
            self.start.get_bottom(),
            self.end.get_top(),
            *args,
            dashed_ratio=0.1,
            color="white",
            **kwargs,
        )


class Weight(Line):
    def __init__(
        self,
        start: Neuron,
        end: Neuron,
        raw_weight: float | None,
        *args,
        **kwargs,
    ):
        self.start = start
        self.end = end
        self.raw_weight = raw_weight
        super().__init__(
            self.start.get_right(),
            self.end.get_left(),
            *args,
            stroke_width=1,
            **kwargs,
        )


class Layer(VGroup):
    def __init__(
        self,
        number,
        indexes: List = None,
        buff=0.5,
        neuron_kwargs: Dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.connections: Dict[Tuple[Neuron, Neuron], Weight] = {}
        self.weights = VGroup()
        if neuron_kwargs is None:
            neuron_kwargs = {}

        self.indexes = list(range(number)) if indexes is None else indexes
        self.neurons = [Neuron(**neuron_kwargs) for _ in self.indexes]

        # the first neuron should always be added without an ellipsis
        previous_idx = self.indexes[0]
        previous_neuron = self.neurons[0]
        self.add(previous_neuron)

        for idx, neuron in zip(self.indexes[1:], self.neurons[1:]):
            if previous_idx != idx - 1:
                # add ...
                dash = EllipsisBetween(
                    neuron,
                    previous_neuron,
                )
                self.add(dash)
            self.add(neuron)
            previous_idx = idx
            previous_neuron = neuron

        self.buff = buff
        self.arrange(direction=DOWN, buff=self.buff)

    def get_scale_factor(self, height=None):
        if height is None:
            height = config.frame_height
        return (height - self.buff) / self.height

    def scale(self, height=None, **kwargs) -> Self:
        return super().scale(self.get_scale_factor(height))

    def connect_back(
        self,
        previous_layer: Self,
        raw_weights: np.array,
        **kwargs,
    ):
        self.weights = VGroup()
        self.connections.clear()

        raw_weights = raw_weights[np.ix_(previous_layer.indexes, self.indexes)]

        for (row, col), raw_weight in np.ndenumerate(raw_weights):
            back_neuron: Neuron = previous_layer.neurons[row]
            current_neuron: Neuron = self.neurons[col]
            weight = Weight(
                back_neuron,
                current_neuron,
                raw_weight,
                **kwargs,
            )
            self.weights.add(weight)
            self.connections[(back_neuron, current_neuron)] = weight

        return self.weights

    def apply_to_weights(self, previous_layer, raw_weights, func):
        raw_weights = raw_weights[np.ix_(previous_layer.indexes, self.indexes)]

        for (row, col), raw_weight in np.ndenumerate(raw_weights):
            back_neuron: Neuron = previous_layer.neurons[row]
            current_neuron: Neuron = self.neurons[col]
            weight = self.connections[(back_neuron, current_neuron)]
            func(weight, raw_weight)

    def update_connections(
        self,
        previous_layer: Self,
        raw_weights: np.array,
    ):
        # noinspection PyShadowingNames
        def update(weight, raw_weight):
            weight.raw_weight = raw_weight

        self.apply_to_weights(previous_layer, raw_weights, update)

        return self.weights


