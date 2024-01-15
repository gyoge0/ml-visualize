from manim import *

from ..visualize.neuron import Neuron


class NeuronTesting(Scene):
    def construct(self):
        neuron = Neuron()
        self.add(neuron)


if __name__ == "__main__":
    NeuronTesting().construct()
