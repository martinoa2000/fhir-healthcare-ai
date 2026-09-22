"""Clinical feature layer: patient records in, model-ready feature vectors out."""

from fhir_healthcare_ai.features.builder import (
    FeatureBuilder,
    FeatureSet,
    build_feature_frame,
    describe_features,
)

__all__ = ["FeatureBuilder", "FeatureSet", "build_feature_frame", "describe_features"]
