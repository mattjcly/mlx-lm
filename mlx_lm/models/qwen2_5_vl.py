# Copyright © 2025 Apple Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from . import qwen2
from .qwen2 import ModelArgs


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        if args.rope_scaling.get("type", None) is not None:
            if args.rope_scaling["type"] == "mrope":
                args.rope_scaling["type"] = "default"
        self.language_model = qwen2.Model(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        return self.language_model(
            inputs, cache=cache, mask=mask, input_embeddings=input_embeddings
        )

    def sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))
        weights.pop("vision_tower", None)
        weights.pop("multi_modal_projector", None)
        lm_weights = dict(tree_flatten(weights["language_model"]))
        lm_weights = self.language_model.sanitize(lm_weights)
        weights["language_model"] = tree_unflatten(list(lm_weights.items()))
        return dict(tree_flatten(weights))

    @property
    def layers(self):
        return self.language_model.model.layers
