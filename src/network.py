from typing import TypeVar
from dataclasses import dataclass

from manim import *
from animations import *
from layer import *
from tensorflow import keras
from tensorflow.keras import models

from weights import Weights


class Indexes:
    pass


@dataclass
class FirstLastIdx(Indexes):
    count: int


class AllIdx(Indexes):
    pass


T = TypeVar("T")


def _verify_length_or_repeat(item: T | List[T], other: List, item_name: str) -> List[T]:
    if isinstance(item, list) and len(item) < len(other):
        raise ValueError(f"Expected {item_name} to be at least length {len(other)}")
    elif isinstance(item, list):
        return item
    else:
        return [item for _ in other]


def _if_none(item: T | None, other: T) -> T:
    return other if item is None else item


def _get_indexes(layer: tf.keras.layers.Layer, indexes: Indexes):
    output_shape = layer.output_shape[1]
    match indexes:
        case FirstLastIdx(count) if count * 2 <= output_shape:
            return [*range(0, count), *range(output_shape - count, output_shape)]
        case _:
            return list(range(output_shape))


def create_layers(
    model: models.Model,
    max_neurons: int | list[int] = 8,
    indexes: Indexes | list[Indexes] | None = None,
    layer_buff: float = 0.5,
    scale: bool | float = False,
    neuron_kwargs: Dict = None,
    layer_kwargs: Dict = None,
) -> List[Layer]:
    max_neurons = _verify_length_or_repeat(max_neurons, model.layers, "max_neurons")
    # noinspection PyTypeChecker
    indexes = _if_none(indexes, AllIdx)
    indexes = _verify_length_or_repeat(indexes, model.layers, "indexes")
    neuron_kwargs = _if_none(neuron_kwargs, {})
    layer_kwargs = _if_none(layer_kwargs, {})

    layers = []

    for layer, layer_max, layer_idx in zip(model.layers, max_neurons, indexes):
        layer_actual_indexes = _get_indexes(layer, layer_idx)
        current_layer = Layer(
            # if indexes are specified, we shouldn't need number?
            number=layer_max,
            indexes=layer_actual_indexes,
            buff=layer_buff,
            neuron_kwargs={
                "radius": 0.125,
                "color": WHITE,
                **neuron_kwargs,
            },
            **layer_kwargs,
        )
        layers.append(current_layer)

        if scale is True:  # True passed in so use default
            current_layer.scale()
        elif scale:  # number passed in so use scale
            current_layer.scale(scale)

    return layers


def create_weights(
    model: models.Model,
    layers: List[Layer],
    weight_kwargs: Dict = None,
    weight_group_kwargs: Dict = None,
    color_weights: bool = False,
    negative: ManimColor = PURE_BLUE,
    positive: ManimColor = PURE_RED,
    zero: ManimColor = WHITE,
):
    weight_kwargs = _if_none(weight_kwargs, {})
    weight_group_kwargs = _if_none(weight_group_kwargs, {})

    mweights = []
    for keras_layer, current_mlayer, last_mlayer in zip(
        model.layers[1:],
        layers[1:],
        layers[:-1],
    ):
        if not isinstance(keras_layer, keras.layers.Dense):
            continue
        keras_layer: layers.Dense = keras_layer
        raw_weights = keras_layer.weights[0]
        mweight = Weights(
            last_mlayer,
            current_mlayer,
            raw_weights.numpy(),
            weight_kwargs,
            **weight_group_kwargs,
        )
        if color_weights:
            mweight.color_weights(negative, positive, zero)
        mweights.append(mweight)

    mweights = VGroup(*mweights)
    return mweights
