"""Inference preprocessing for twanghcmut's joint four-suite N1.5 checkpoint."""

from copy import deepcopy

from examples.Libero.custom_data_config import LiberoDataConfig
from gr00t.model.transforms import GR00TTransform


class LiberoCombinedDataConfig(LiberoDataConfig):
    """Keep LIBERO min/max transforms while disabling checkpoint-incompatible tiling."""

    def transform(self):
        composed = super().transform()
        model_transforms = [
            transform for transform in composed.transforms
            if isinstance(transform, GR00TTransform)
        ]
        if len(model_transforms) != 1:
            raise RuntimeError("Expected exactly one GR00TTransform in LIBERO preprocessing")
        model_transform = model_transforms[0]
        processor = deepcopy(model_transform.eagle_processor)
        processor.image_processor.min_dynamic_tiles = 1
        processor.image_processor.max_dynamic_tiles = 1
        processor.image_processor.use_thumbnail = False
        model_transform.eagle_processor = processor
        return composed
