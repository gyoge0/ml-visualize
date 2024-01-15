from typing import Tuple

from manim import *
from typing_extensions import Self
from layer import *
import tensorflow as tf
import numpy as np


def get_color_on_gradient(
    value,
    max_value,
    negative=PURE_BLUE,
    zero=WHITE,
    positive=PURE_RED,
):
    normalized_value = value / max_value

    if normalized_value < 0:
        new_color = zero.interpolate(negative, alpha=np.abs(normalized_value))
    else:
        new_color = zero.interpolate(positive, alpha=normalized_value)

    return new_color


class Weights(VGroup):
    def __init__(
        self,
        previous_layer: Layer,
        current_layer: Layer,
        raw_weights: np.array,
        weight_kwargs: Dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.connections: Dict[Tuple[Neuron, Neuron], Weight] = {}
        self.previous_layer: Layer = previous_layer
        self.current_layer: Layer = current_layer

        self.raw_weights: np.array = None
        self.set_weights(raw_weights)

        weight_kwargs = {} if weight_kwargs is None else weight_kwargs
        self.create_connections(**weight_kwargs)

    def set_weights(self, raw_weights):
        self.raw_weights = raw_weights[
            np.ix_(self.previous_layer.indexes, self.current_layer.indexes)
        ].copy()

    def create_connections(self, **kwargs):
        self.connections.clear()
        self.submobjects.clear()

        for (row, col), raw_weight in np.ndenumerate(self.raw_weights):
            previous_neuron: Neuron = self.previous_layer.neurons[row]
            current_neuron: Neuron = self.current_layer.neurons[col]
            weight = Weight(
                previous_neuron,
                current_neuron,
                raw_weight,
                **kwargs,
            )
            self.add(weight)
            self.connections[(previous_neuron, current_neuron)] = weight

        return self

    def apply_to_connections(self, func):
        for (row, col), raw_weight in np.ndenumerate(self.raw_weights):
            back_neuron: Neuron = self.previous_layer.neurons[row]
            current_neuron: Neuron = self.current_layer.neurons[col]
            weight = self.connections[(back_neuron, current_neuron)]
            func(weight, raw_weight)

    def update_connection_weights(self):
        # noinspection PyShadowingNames
        def update(weight, raw_weight):
            weight.raw_weight = raw_weight

        self.apply_to_connections(update)

        return self

    def color_weights(
        self,
        negative: ManimColor = PURE_BLUE,
        positive: ManimColor = PURE_RED,
        zero: ManimColor = BLACK,
    ) -> Self:
        max_weight = np.abs(self.raw_weights).max()

        def update(weight, raw_weight):
            new_color = get_color_on_gradient(
                raw_weight, max_weight, negative, zero, positive
            )
            weight.set_color(new_color)

        self.apply_to_connections(update)
        return self

    def animate_color_weights(
        self,
        negative: ManimColor = PURE_BLUE,
        positive: ManimColor = PURE_RED,
        zero: ManimColor = BLACK,
        **kwargs,
    ) -> AnimationGroup:
        max_weight = np.abs(self.raw_weights).max()
        animations = []

        def update(weight, raw_weight):
            new_color = get_color_on_gradient(
                raw_weight, max_weight, negative, zero, positive
            )
            animations.append(weight.animate.set_color(new_color))

        self.apply_to_connections(update)
        return AnimationGroup(*animations, **kwargs)
