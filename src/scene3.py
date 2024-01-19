from tensorflow.keras import models

from network_graph import *

# model = models.Sequential(
#     [
#         layers.Dense(8),
#         layers.Dense(8),
#     ]
# )
# model.compile(
#     optimizer="adam",
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# )
# model.fit([[0 for _ in range(8)]], [[0]])
# model.save(r"D:\ml-visualize\model")
model = models.load_model(r"D:\ml-visualize\model")


class TestNeuron(Scene):
    def construct(self):
        nn = NeuralNetwork(
            model,
            (4, 4),
            4,
        )
        self.play(Create(nn))
        self.pause(1)


if __name__ == "__main__":
    TestNeuron().construct()
