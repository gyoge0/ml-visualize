from ..visualize.animations import (
    CreateGroup,
    CreateWeightsWithColor,
    AnimateWeightColors,
    CreateGroupCascade,
)
from ..visualize.layer import *


class CreateNeuralNetwork(Scene):
    def construct(self):
        first_layer = (
            Layer(
                24,
                indexes=[*range(0, 12), *range(784 - 12, 784)],
                buff=1,
                neuron_kwargs={
                    "color": "white",
                    "stroke_width": DEFAULT_STROKE_WIDTH / 2,
                },
            )
            .scale()
            .shift(LEFT * 2)
        )

        second_layer = Layer(
            16,
            indexes=[*range(0, 4), *range(16 - 4, 16)],
            buff=3,
            neuron_kwargs={
                "color": "white",
                "stroke_width": DEFAULT_STROKE_WIDTH / 2,
            },
        ).scale()

        third_layer = (
            Layer(
                16,
                indexes=[*range(0, 4), *range(16 - 4, 16)],
                buff=3,
                neuron_kwargs={
                    "color": "white",
                    "stroke_width": DEFAULT_STROKE_WIDTH / 2,
                },
            )
            .scale()
            .shift(RIGHT * 2)
        )

        layers = VGroup(first_layer, second_layer, third_layer)
        layers.arrange(RIGHT, buff=1)
        self.play(CreateGroupCascade(layers, lag_ratio=0.2))

        raw_weights_second_first = np.load(r"D:\ml-visualize\weights\second_first.npy")
        weights_second_first = second_layer.connect_back(
            first_layer, raw_weights_second_first
        )

        raw_weights_third_second = np.load(r"D:\ml-visualize\weights\third_second.npy")
        weights_third_second = third_layer.connect_back(
            second_layer, raw_weights_third_second
        )
        self.play(CreateGroup(weights_second_first, lag_ratio=0.005))
        self.play(CreateGroup(weights_third_second, lag_ratio=0.005))

        a1 = AnimateWeightColors(
            second_layer,
            first_layer,
            raw_weights_second_first,
            zero=WHITE,
            lag_ratio=0.005,
        )
        self.play(a1)

        a2 = AnimateWeightColors(
            third_layer,
            second_layer,
            raw_weights_third_second,
            zero=WHITE,
            lag_ratio=0.02,
        )
        self.play(a2)
        self.pause(3)


# for debugging
if __name__ == "__main__":
    CreateNeuralNetwork().render()
