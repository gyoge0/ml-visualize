from network import *
from animations import *
from manim import *


class CreateNeuralNetwork(Scene):
    def construct(self):
        model = models.load_model(r"model")

        layers: List[Layer] = create_layers(
            model,
            max_neurons=[8, 8, 1],
            indexes=[
                FirstLastIdx(3),
                AllIdx(),
                FirstLastIdx(2),
            ],
            scale=True,
        )

        layers[0].shift(LEFT * 2)
        layers[2].shift(RIGHT * 2)

        weights = create_weights(
            model,
            layers=layers,
            color_weights=True,
            zero=LIGHT_GRAY,
        )

        self.play(CreateGroupCascade(layers, lag_ratio=0.025))
        self.play(CreateGroupCascade(weights, lag_ratio=0.01))
        self.pause(3)


if __name__ == "__main__":
    CreateNeuralNetwork().render()
