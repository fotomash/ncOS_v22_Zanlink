# poi_quality_predictor.py
# Scores Points of Interest (POIs) based on structural and contextual features

import numpy as np

class POIQualityPredictor:
    def __init__(self, model=None):
        self.model = model  # Optional ML model (sklearn or compatible)

    def extract_features(self, poi_dict):
        """
        Input Example:
        {
            "mitigation_count": 1,
            "time_in_range": 18,
            "distance_to_eqh": 4.2,
            "volume_weighted_score": 0.84,
            "overlap_with_fvg": 1
        }
        """
        return np.array([
            poi_dict.get("mitigation_count", 0),
            poi_dict.get("time_in_range", 0),
            poi_dict.get("distance_to_eqh", 0),
            poi_dict.get("volume_weighted_score", 0),
            poi_dict.get("overlap_with_fvg", 0)
        ]).reshape(1, -1)

    def predict_quality(self, poi_dict):
        if not self.model:
            return 0.5  # neutral fallback
        X = self.extract_features(poi_dict)
        return self.model.predict_proba(X)[0][1]  # probability it's high quality


if __name__ == "__main__":
    predictor = POIQualityPredictor()
    sample_poi = {
        "mitigation_count": 2,
        "time_in_range": 12,
        "distance_to_eqh": 3.4,
        "volume_weighted_score": 0.9,
        "overlap_with_fvg": 1
    }
    quality = predictor.predict_quality(sample_poi)
    print(f"POI Quality Score: {quality:.2f}")
