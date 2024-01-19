from manim import *


class Neuron(Circle):
    def __init__(
        self,
        text: float | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.text = Text("" if Text is None else str(text))
        self.text.align_to(self.get_center())
        self.submobjects.append(self.text)
