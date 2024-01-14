from manim import *
from typing_extensions import Self
import numpy as np


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

        if neuron_kwargs is None:
            neuron_kwargs = {}

        self.indexes = list(range(number)) if indexes is None else indexes
        self.neurons = [Circle(**neuron_kwargs) for _ in self.indexes]

        # the first neuron should always be added without an ellipsis
        previous_idx = self.indexes[0]
        previous_neuron = self.neurons[0]
        self.add(previous_neuron)

        # go in reverse order so bottom to top
        for idx, neuron in zip(self.indexes[1:], self.neurons[1:]):
            if previous_idx != idx - 1:
                # add ...
                dash = DashedLine(
                    neuron.get_top() + UP * 0.1,
                    previous_neuron.get_bottom() + DOWN * 0.1,
                    dashed_ratio=0.1,
                    color="white",
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
        weights: np.array,
        **kwargs,
    ) -> List[Line]:
        """
        Return a set of lines representing the weights between this layer and a previous layer.

        Weights should be a numpy array where each row has weights from a single neuron in the previous layer,
        and each column has weights to a single neuron in the current layer.

        :param previous_layer:
        :param weights:
        """

        back_indexes = previous_layer.indexes
        back_neurons = previous_layer.neurons
        self_indexes = self.indexes
        self_neurons = self.neurons

        weights = weights[np.ix_(back_indexes, self_indexes)]
        lines = []

        for (row, col), weight in np.ndenumerate(weights):
            back_neuron: Circle = back_neurons[row]
            self_neuron: Circle = self_neurons[col]
            lines.append(
                Line(
                    back_neuron.get_right(),
                    self_neuron.get_left(),
                    stroke_width=1,
                    **kwargs,
                )
            )

        return lines


class CreateLayer(AnimationGroup):
    def __init__(self, layer, **kwargs):
        animations = [Create(neuron) for neuron in layer]
        super().__init__(*animations, **kwargs)


class CreateNeuralNetwork(Scene):
    def construct(self):
        first_layer = Layer(8, buff=1).scale().shift(LEFT * 2)
        second_layer = (
            Layer(
                8,
                indexes=[*range(0, 2), *range(10, 16)],
                buff=1,
                neuron_kwargs={"color": "blue", "radius": 0.5},
            )
            .scale()
            .shift(RIGHT * 2)
        )

        self.play(CreateLayer(first_layer, lag_ratio=0.1))
        self.play(CreateLayer(second_layer, lag_ratio=0.1))

        weights = np.genfromtxt(
            r"D:\ml-visualize\weights.csv", delimiter=",", dtype=float
        )
        first_to_second = second_layer.connect_back(first_layer, weights)
        self.play(*(Create(line) for line in first_to_second))
        self.pause(3)


# for debugging
if __name__ == "__main__":
    CreateNeuralNetwork().render()
