def get_extractor(extractor_name: str,
                  n_features: int = 10000):
    if extractor_name == "sift":
        from feature_extractor import SiftExtractor
        return SiftExtractor(n_features)
    elif extractor_name == "root_sift":
        from feature_extractor import RootSiftExtractor
        return RootSiftExtractor(n_features)
    elif extractor_name == "tomasi":
        from feature_extractor import GoodFeaturesToTrackExtractor
        return GoodFeaturesToTrackExtractor(n_features)
    elif extractor_name == "orb":
        from feature_extractor import OrbExtractor
        return OrbExtractor(n_features)
    else:
        raise NotImplementedError("Extractor not implemented")


def get_matcher(matcher_name: str,
                n_ransac_max_iters: int,
                ransac_confidence: float,
                ransac_reproj_threshold: float = 3.0):
    if matcher_name == "standard":
        from matcher import FeatureMatcher
        return FeatureMatcher(fm_ransac_confidence=ransac_confidence,
                              fm_ransac_max_iters=n_ransac_max_iters,
                              fm_ransac_reproj_threshold=ransac_reproj_threshold)
    else:
        raise NotImplementedError("Matcher not implemented")