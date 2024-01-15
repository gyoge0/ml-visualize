from typing import Iterable

from manim import *

from layer import Layer


class CreateGroup(AnimationGroup):
    def __init__(self, group: Iterable[VMobject], **kwargs):
        animations = (Create(item) for item in group)
        super().__init__(*animations, **kwargs)


class CreateGroupCascade(AnimationGroup):
    def __init__(self, group: Iterable[VMobject], **kwargs):
        animations = []

        # recursively add all sub items
        # noinspection PyShadowingNames
        def add_items(group):
            for item in group:
                if isinstance(item, VGroup):
                    add_items(item)
                else:
                    animations.append(Create(item))
        add_items(group)

        super().__init__(*animations, **kwargs)


class AnimateWeightColors(AnimationGroup):
    def __init__(
        self,
        current_layer: Layer,
        previous_layer: Layer,
        raw_weights: np.array,
        negative=PURE_BLUE,
        zero=WHITE,
        positive=PURE_RED,
        **kwargs,
    ):
        animations = []

        max_weight = np.abs(raw_weights).max()

        def update(weight, raw_weight):
            new_color = get_color_on_gradient(
                raw_weight, max_weight, negative, zero, positive
            )
            animations.append(weight.animate.set_color(new_color))

        current_layer.apply_to_weights(previous_layer, raw_weights, update)

        super().__init__(*animations, **kwargs)


class CreateWeightsWithColor(AnimationGroup):
    def __init__(
        self,
        current_layer: Layer,
        previous_layer: Layer,
        raw_weights: np.array,
        negative=PURE_BLUE,
        zero=WHITE,
        positive=PURE_RED,
        **kwargs,
    ):
        animations = []

        max_weight = np.abs(raw_weights).max()

        def update(weight, raw_weight):
            new_color = get_color_on_gradient(
                raw_weight, max_weight, negative, zero, positive
            )
            weight.set_color(new_color)
            animations.append(Create(weight))

        current_layer.apply_to_weights(previous_layer, raw_weights, update)

        super().__init__(*animations, **kwargs)
