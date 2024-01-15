from typing import Tuple

from manim import *
from typing_extensions import Self
import tensorflow as tf
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
                    previous_neuron,
                    neuron,
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
        return min((height - self.buff) / self.height, 1)

    def scale(self, height=None, **kwargs) -> Self:
        for item in self:
            item.stroke_width = DEFAULT_STROKE_WIDTH * self.get_scale_factor(height)
        return super().scale(self.get_scale_factor(height))
